import argparse
import os
from td3_agent_CarRacing import CarRacingTD3Agent


def parse_args():
	parser = argparse.ArgumentParser(description="TD3 on CarRacing-v2 (with ablation switches)")
	parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log dir (use a different one per experiment)")
	parser.add_argument("--no-twin", action="store_true", help="ablation 1: use a single Q-network")
	parser.add_argument("--no-smoothing", action="store_true", help="ablation 2: disable target policy smoothing")
	parser.add_argument("--update-freq", type=int, default=None, help="ablation 3: delayed update steps (actor + target nets)")
	parser.add_argument("--sigma-max", type=float, default=None, help="ablation 4: max exploration noise sigma")
	parser.add_argument("--sigma-min", type=float, default=None, help="lower bound of the exploration noise sigma")
	parser.add_argument("--sigma-decay-steps", type=int, default=None, help="decay sigma linearly from sigma_max to sigma_min over this many steps")
	parser.add_argument("--reward-mode", type=str, default=None, choices=["original", "custom", "custom_b", "custom_bc"], help="ablation 5: reward function")
	parser.add_argument("--speed-safe", type=float, default=None, help="custom_bc: speed above which steering hard is penalized")
	parser.add_argument("--speed-coef", type=float, default=None, help="custom_bc: coefficient of the speed penalty")
	parser.add_argument("--brake-final", type=float, default=None, help="limit curriculum: brake_rate reached at --ramp-end (start 0.015, i.e. max brake 2*rate)")
	parser.add_argument("--gas-floor-final", type=float, default=None, help="limit curriculum: gas floor reached at --ramp-end (start 0.1)")
	parser.add_argument("--ramp-start", type=int, default=None, help="limit curriculum: step at which the limits start to relax")
	parser.add_argument("--ramp-end", type=int, default=None, help="limit curriculum: step at which the limits are fully relaxed")
	parser.add_argument("--stuck-speed", type=float, default=None, help="stuck rule: end the episode when the speed stays below this value (training env only)")
	parser.add_argument("--stuck-steps", type=int, default=None, help="stuck rule: number of consecutive slow steps that trigger it (default 50)")
	parser.add_argument("--stuck-penalty", type=float, default=None, help="stuck rule: reward of the step that ends the episode (default -100, same as leaving the road)")
	parser.add_argument("--buffer", type=int, default=None, help="replay buffer capacity (kept on the GPU; about 74 KB per transition)")
	parser.add_argument("--min-buffer", type=int, default=None, help="do not update before the buffer holds this many transitions")
	parser.add_argument("--eval-interval", type=int, default=None, help="evaluate (and save a checkpoint) every N training episodes")
	parser.add_argument("--eval-episodes", type=int, default=None, help="number of games per evaluation")
	parser.add_argument("--resume", action="store_true", help="continue from <logdir>/state.pth if it exists")
	parser.add_argument("--episodes", type=int, default=None, help="override total_episode (also stretches the sigma decay)")
	parser.add_argument("--steps", type=int, default=None, help="stop training after this many environment steps (checked after each episode)")
	return parser.parse_args()


if __name__ == '__main__':
	# default hyperparameters
	config = {
		"gpu": True,
		"training_steps": 1e8,
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": 100000,
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 5000,
		"logdir": 'log/CarRacing/td3_test/',
		"update_freq": 2,
		"eval_interval": 10,
		"eval_episode": 10,
		# ablation switches (defaults = full TD3)
		"twin_q": True,
		"target_smoothing": True,
		"smooth_sigma": 0.2,
		"smooth_clip": 0.5,
		"sigma_max": 0.1,
		"sigma_min": 0.01,
		"reward_mode": "original",
		"speed_safe": None,
		"speed_coef": 0.5,
		# brake / gas limit curriculum (defaults = original limits, i.e. no relaxation)
		"brake_final": 0.015,
		"gas_floor_final": 0.1,
		"ramp_start": 0,
		"ramp_end": 1,
		# stuck rule (default off) and replay-buffer refill after a resume
		"stuck_speed": None,
		"stuck_steps": 50,
		"stuck_penalty": -100.0,
	}

	# command-line overrides, only applied when the flag is given
	args = parse_args()
	if args.logdir is not None:
		config["logdir"] = args.logdir
	if args.no_twin:
		config["twin_q"] = False
	if args.no_smoothing:
		config["target_smoothing"] = False
	if args.update_freq is not None:
		config["update_freq"] = args.update_freq
	if args.sigma_max is not None:
		config["sigma_max"] = args.sigma_max
	if args.sigma_min is not None:
		config["sigma_min"] = args.sigma_min
	if args.sigma_decay_steps is not None:
		config["sigma_decay_steps"] = args.sigma_decay_steps
	if args.reward_mode is not None:
		config["reward_mode"] = args.reward_mode
	if args.speed_safe is not None:
		config["speed_safe"] = args.speed_safe
	if args.speed_coef is not None:
		config["speed_coef"] = args.speed_coef
	for flag, key in (("brake_final", "brake_final"), ("gas_floor_final", "gas_floor_final"),
	                  ("ramp_start", "ramp_start"), ("ramp_end", "ramp_end")):
		if getattr(args, flag) is not None:
			config[key] = getattr(args, flag)
	for flag, key in (("stuck_speed", "stuck_speed"), ("stuck_steps", "stuck_steps"), ("stuck_penalty", "stuck_penalty"),
	                  ("buffer", "replay_buffer_capacity"), ("min_buffer", "min_buffer"),
	                  ("eval_interval", "eval_interval"), ("eval_episodes", "eval_episode")):
		if getattr(args, flag) is not None:
			config[key] = getattr(args, flag)
	state_path = os.path.join(config["logdir"], "state.pth")
	resuming = args.resume and os.path.exists(state_path)
	if resuming and "min_buffer" not in config:
		config["min_buffer"] = 5000  # the buffer is empty after a resume: refill it before updating again
	if args.episodes is not None:
		config["total_episode"] = args.episodes
	if args.steps is not None:
		config["training_steps"] = args.steps

	agent = CarRacingTD3Agent(config)
	start_episode = 0
	if args.resume:
		if resuming:
			start_episode = agent.load_state(state_path)
			print(f"Resumed from {state_path}: step {agent.total_time_step}, episode {start_episode}")
		else:
			print(f"--resume given but {state_path} does not exist: starting from scratch")
	agent.train(start_episode=start_episode)
