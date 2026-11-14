import argparse
from td3_agent_CarRacing import CarRacingTD3Agent


def parse_args():
	parser = argparse.ArgumentParser(description="TD3 on CarRacing-v2 (with ablation switches)")
	parser.add_argument("--logdir", type=str, default=None, help="TensorBoard log dir (use a different one per experiment)")
	parser.add_argument("--no-twin", action="store_true", help="ablation 1: use a single Q-network")
	parser.add_argument("--no-smoothing", action="store_true", help="ablation 2: disable target policy smoothing")
	parser.add_argument("--update-freq", type=int, default=None, help="ablation 3: delayed update steps (actor + target nets)")
	parser.add_argument("--sigma-max", type=float, default=None, help="ablation 4: max exploration noise sigma")
	parser.add_argument("--reward-mode", type=str, default=None, choices=["original", "custom", "custom_b", "custom_bc"], help="ablation 5: reward function")
	parser.add_argument("--speed-safe", type=float, default=None, help="custom_bc: speed above which steering hard is penalized")
	parser.add_argument("--speed-coef", type=float, default=None, help="custom_bc: coefficient of the speed penalty")
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
	if args.reward_mode is not None:
		config["reward_mode"] = args.reward_mode
	if args.speed_safe is not None:
		config["speed_safe"] = args.speed_safe
	if args.speed_coef is not None:
		config["speed_coef"] = args.speed_coef
	if args.episodes is not None:
		config["total_episode"] = args.episodes
	if args.steps is not None:
		config["training_steps"] = args.steps

	agent = CarRacingTD3Agent(config)
	agent.train()
