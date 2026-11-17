import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from replay_buffer.replay_buffer import ReplayMemory
from abc import ABC, abstractmethod


class GaussianNoise:
	def __init__(self, dim, mu=None, std=None):
		self.mu = mu if mu else np.zeros(dim)
		self.std = np.ones(dim) * std if std else np.ones(dim) * .1
	
	def reset(self):
		pass

	def generate(self):
		return np.random.normal(self.mu, self.std)

class OUNoiseGenerator:
    def __init__(self, mean, std_dev, theta=0.3, dt=5e-2):
        self.theta = theta
        self.dt = dt
        self.mean = mean
        self.std_dev = std_dev

        self.x = None

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.mean.shape)

    def generate(self):
        self.x = (self.x
                  + self.theta * (self.mean - self.x) * self.dt
                  + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))

        return self.x

class TD3BaseAgent(ABC):
	def __init__(self, config):
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.batch_size = int(config["batch_size"])
		self.warmup_steps = config["warmup_steps"]
		self.total_episode = config["total_episode"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]
		self.gamma = config["gamma"]
		self.tau = config["tau"]
		self.update_freq = config["update_freq"]

		# exploration noise schedule (ablation: action noise level)
		self.sigma_max = config.get("sigma_max", 0.1)
		self.sigma_min = config.get("sigma_min", 0.01)
		# if set, sigma decays linearly from sigma_max to sigma_min over this many environment steps
		# (default None: the original per-episode formula, which hardly decays when total_episode is huge)
		self.sigma_decay_steps = config.get("sigma_decay_steps")

		# the replay buffer lives on the training device (GPU memory when a GPU is used)
		self.replay_buffer = ReplayMemory(int(config["replay_buffer_capacity"]), device=self.device)
		# do not start updating before the buffer holds at least this many transitions (matters after a resume,
		# when the buffer starts empty but the warm-up is already over)
		self.min_buffer = max(int(config.get("min_buffer", self.batch_size)), self.batch_size)
		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, state, sigma=0.0):
		# based on the behavior (actor) network and exploration noise
		
		return NotImplementedError
				
	
	def update(self):
		# update the behavior networks
		self.update_behavior_network()
		# update the target networks
		if self.total_time_step % self.update_freq == 0:
			self.update_target_network(self.target_actor_net, self.actor_net, self.tau)
			self.update_target_network(self.target_critic_net1, self.critic_net1, self.tau)
			self.update_target_network(self.target_critic_net2, self.critic_net2, self.tau)

	@abstractmethod
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		# calculate the loss and update the behavior network

		return NotImplementedError
		

	@staticmethod
	def update_target_network(target_net, net, tau):
		# update target network by "soft" copying from behavior network
		for target, behavior in zip(target_net.parameters(), net.parameters()):
			target.data.copy_((1 - tau) * target.data + tau * behavior.data)
	
	def exploration_sigma(self, episode):
		if self.sigma_decay_steps:
			frac = min(1.0, self.total_time_step / self.sigma_decay_steps)
			return max(self.sigma_max * (1 - frac), self.sigma_min)
		return max(self.sigma_max * (1 - episode / self.total_episode), self.sigma_min)

	def train(self, start_episode=0):
		for episode in range(start_episode, self.total_episode):
			total_reward = 0
			state, infos = self.env.reset()
			self.noise.reset()
			for t in range(10000):
				if self.total_time_step < self.warmup_steps:
					action = self.env.action_space.sample()
				else:
					# exploration degree
					#sigma = max(0.1*(1-episode/self.total_episode), 0.01)
					sigma = self.exploration_sigma(episode)
					action = self.decide_agent_actions(state, sigma=sigma)
				
				next_state, reward, terminates, truncates, _ = self.env.step(action)
				self.replay_buffer.append(state, action, [reward/10], next_state, [int(terminates)])
				if self.total_time_step >= self.warmup_steps and len(self.replay_buffer) >= self.min_buffer:
					self.update()

				self.total_time_step += 1
				total_reward += reward
				state = next_state
				if terminates or truncates:
					self.writer.add_scalar('Train/Episode Reward', total_reward, self.total_time_step)
					print(
						'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(self.total_time_step, episode+1, t, total_reward))
				
					break
			
			if (episode+1) % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)
				# full training state (networks, optimizers, step counter) so that a run can be resumed
				self.save_state(os.path.join(self.writer.log_dir, "state.pth"), episode + 1)

			# stop once the step budget is used up (training_steps defaults to 1e8, i.e. effectively never)
			if self.total_time_step >= self.training_steps:
				break

		self.writer.close()

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for episode in range(self.eval_episode):
			total_reward = 0
			state, infos = self.test_env.reset()
			for t in range(10000):
				action = self.decide_agent_actions(state)
				next_state, reward, terminates, truncates, _ = self.test_env.step(action)
				total_reward += reward
				state = next_state
				if terminates or truncates:
					print(
						'Episode: {}\tLength: {:3d}\tTotal reward: {:.2f}'
						.format(episode+1, t, total_reward))
					all_rewards.append(total_reward)
					break

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(
				{
					'actor': self.actor_net.state_dict(),
					'critic1': self.critic_net1.state_dict(),
					'critic2': self.critic_net2.state_dict(),
					# action limits (brake_rate, gas_floor) in force when the checkpoint was saved, if the agent has them
					'limits': tuple(self.limits()) if hasattr(self, 'limits') else None,
				}, save_path)

	# load model
	def load(self, load_path):
		checkpoint = torch.load(load_path)
		self.actor_net.load_state_dict(checkpoint['actor'])
		self.critic_net1.load_state_dict(checkpoint['critic1'])
		self.critic_net2.load_state_dict(checkpoint['critic2'])
		# old checkpoints have no 'limits' -> the original limits are used
		if checkpoint.get('limits') is not None and hasattr(self, 'fixed_limits'):
			self.fixed_limits = tuple(checkpoint['limits'])

	# save everything needed to continue training later (networks, targets, optimizers, step counter)
	STATE_KEYS = ('actor_net', 'critic_net1', 'critic_net2', 'target_actor_net', 'target_critic_net1',
	              'target_critic_net2')
	OPT_KEYS = ('actor_opt', 'critic_opt1', 'critic_opt2')

	def save_state(self, path, next_episode):
		state = {k: getattr(self, k).state_dict() for k in self.STATE_KEYS}
		state.update({k: getattr(self, k).state_dict() for k in self.OPT_KEYS if hasattr(self, k)})
		state['total_time_step'] = self.total_time_step
		state['next_episode'] = next_episode
		torch.save(state, path + '.tmp')
		os.replace(path + '.tmp', path)  # atomic: a disconnect cannot leave a half-written file

	# restore a state saved by save_state(); returns the episode index to continue from
	def load_state(self, path):
		state = torch.load(path, map_location=self.device)
		for k in self.STATE_KEYS:
			getattr(self, k).load_state_dict(state[k])
		for k in self.OPT_KEYS:
			if hasattr(self, k) and k in state:
				getattr(self, k).load_state_dict(state[k])
		self.total_time_step = state['total_time_step']
		return state['next_episode']

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()

