import torch
import torch.nn as nn
import numpy as np
from base_agent import TD3BaseAgent
from models.CarRacing_model import ActorNetSimple, CriticNetSimple
from environment_wrapper.CarRacingEnv import CarRacingEnvironment
import random
from base_agent import OUNoiseGenerator, GaussianNoise

class CarRacingTD3Agent(TD3BaseAgent):
	def __init__(self, config):
		super(CarRacingTD3Agent, self).__init__(config)
		# initialize environment
		self.reward_mode = config.get("reward_mode", "original")
		reward_kwargs = dict(reward_mode=self.reward_mode, speed_safe=config.get("speed_safe"),
		                     speed_coef=config.get("speed_coef", 0.5),
		                     stuck_speed=config.get("stuck_speed"), stuck_steps=config.get("stuck_steps", 50),
		                     stuck_penalty=config.get("stuck_penalty", -100.0))
		self.env = CarRacingEnvironment(N_frame=4, test=False, **reward_kwargs)
		self.test_env = CarRacingEnvironment(N_frame=4, test=True, **reward_kwargs)
		
		# behavior network
		self.actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.actor_net.to(self.device)
		self.critic_net1.to(self.device)
		self.critic_net2.to(self.device)
		# target network
		self.target_actor_net = ActorNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net1 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_critic_net2 = CriticNetSimple(self.env.observation_space.shape[0], self.env.action_space.shape[0], 4)
		self.target_actor_net.to(self.device)
		self.target_critic_net1.to(self.device)
		self.target_critic_net2.to(self.device)
		self.target_actor_net.load_state_dict(self.actor_net.state_dict())
		self.target_critic_net1.load_state_dict(self.critic_net1.state_dict())
		self.target_critic_net2.load_state_dict(self.critic_net2.state_dict())

		# ablation switches
		self.twin_q = config.get("twin_q", True)
		self.target_smoothing = config.get("target_smoothing", True)
		self.smooth_sigma = config.get("smooth_sigma", 0.2)
		self.smooth_clip = config.get("smooth_clip", 0.5)

		# brake / gas limit curriculum: start with the original limits (brake in [0, 2*0.015], gas >= 0.1) and
		# relax them linearly between ramp_start and ramp_end (defaults: finals = starts -> no relaxation)
		self.brake_start = 0.015
		self.gas_floor_start = 0.1
		self.brake_final = config.get("brake_final", self.brake_start)
		self.gas_floor_final = config.get("gas_floor_final", self.gas_floor_start)
		self.ramp_start = config.get("ramp_start", 0)
		self.ramp_end = config.get("ramp_end", 1)
		self.fixed_limits = None  # set by load(): a checkpoint keeps the limits it was saved with

		# set optimizer
		self.lra = config["lra"]
		self.lrc = config["lrc"]
		
		self.actor_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.lra)
		self.critic_opt1 = torch.optim.Adam(self.critic_net1.parameters(), lr=self.lrc)
		self.critic_opt2 = torch.optim.Adam(self.critic_net2.parameters(), lr=self.lrc)

		# choose Gaussian noise or OU noise

		# noise_mean = np.full(self.env.action_space.shape[0], 0.0, np.float32)
		# noise_std = np.full(self.env.action_space.shape[0], 1.0, np.float32)
		# self.noise = OUNoiseGenerator(noise_mean, noise_std)

		self.noise = GaussianNoise(self.env.action_space.shape[0], 0.0, 1.0)
		
	
	def limits(self):
		"""(brake_rate, gas_floor) at the current training step: the original limits until ramp_start, then
		linearly relaxed until ramp_end. A loaded checkpoint keeps the limits it was saved with."""
		if self.fixed_limits is not None:
			return self.fixed_limits
		span = max(1, self.ramp_end - self.ramp_start)
		frac = min(1.0, max(0.0, (self.total_time_step - self.ramp_start) / span))
		return (self.brake_start + frac * (self.brake_final - self.brake_start),
		        self.gas_floor_start + frac * (self.gas_floor_final - self.gas_floor_start))

	def decide_agent_actions(self, state, sigma=0.0, brake_rate=None, gas_floor=None):
		# based on the behavior (actor) network and exploration noise
		cur_brake, cur_gas = self.limits()
		brake_rate = cur_brake if brake_rate is None else brake_rate
		gas_floor = cur_gas if gas_floor is None else gas_floor
		with torch.no_grad():
			# (4, 96, 96) numpy -> (1, 4, 96, 96) float tensor on device
			state = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
			action = self.actor_net(state, brake_rate=brake_rate, gas_floor=gas_floor).squeeze(0).cpu().numpy()

		# exploration noise (sigma = 0 during evaluation -> deterministic)
		if sigma > 0:
			noise = self.noise.generate()
			# brake only lives in [0, 2*brake_rate], so shrink its noise accordingly
			noise[2] *= brake_rate
			action = action + sigma * noise

		# clip to the valid action space: steer [-1,1], gas [0,1], brake [0,1]
		action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
		return action.astype(np.float32)
		

	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		# 1. Clipped Double Q-Learning for Actor-Critic
		# 2. Delayed Policy Updates
		# 3. Target Policy Smoothing Regularization

		# shapes: state (B,4,96,96), action (B,3), reward (B,1), done (B,1)

		brake_rate, gas_floor = self.limits()  # current action limits (same values for actor and target actor)

		## Update Critic ##
		q_value1 = self.critic_net1(state, action)
		q_value2 = self.critic_net2(state, action)
		with torch.no_grad():
			# select action a_next from target actor network
			a_next = self.target_actor_net(next_state, brake_rate=brake_rate, gas_floor=gas_floor)

			# 3. Target Policy Smoothing: add clipped noise to the target action
			if self.target_smoothing:
				noise = torch.randn_like(a_next) * self.smooth_sigma
				noise[:, 2] *= brake_rate  # brake range is small, same reasoning as exploration noise
				noise = noise.clamp(-self.smooth_clip, self.smooth_clip)
				a_next = a_next + noise
				low = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
				high = torch.tensor([1.0, 1.0, 1.0], device=self.device)
				a_next = torch.max(torch.min(a_next, high), low)

			q_next1 = self.target_critic_net1(next_state, a_next)
			q_next2 = self.target_critic_net2(next_state, a_next)
			# 1. Clipped Double Q-learning: take the smaller estimate
			q_next = torch.min(q_next1, q_next2) if self.twin_q else q_next1
			# done=1 only when the episode really terminated, so no bootstrap there
			q_target = reward + self.gamma * (1 - done) * q_next

		# critic loss function
		criterion = nn.MSELoss()
		critic_loss1 = criterion(q_value1, q_target)
		critic_loss2 = criterion(q_value2, q_target)

		# optimize critic
		self.critic_net1.zero_grad()
		critic_loss1.backward()
		self.critic_opt1.step()

		if self.twin_q:
			self.critic_net2.zero_grad()
			critic_loss2.backward()
			self.critic_opt2.step()

		## Delayed Actor(Policy) Updates ##
		# 2. only update the actor every update_freq steps
		if self.total_time_step % self.update_freq == 0:
			# action chosen by the behavior actor (different from the sampled action)
			action = self.actor_net(state, brake_rate=brake_rate, gas_floor=gas_floor)
			# maximize Q  ->  minimize -Q
			actor_loss = -1 * self.critic_net1(state, action).mean()
			# optimize actor
			self.actor_net.zero_grad()
			actor_loss.backward()
			self.actor_opt.step()
		
