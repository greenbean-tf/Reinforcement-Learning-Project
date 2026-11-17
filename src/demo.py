import argparse
import os

import cv2
import numpy as np
from td3_agent_CarRacing import CarRacingTD3Agent

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='path to model_*.pth')
parser.add_argument('--seed', type=int, default=0, help='track seed; episode i uses seed + i')
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--name', default='demo', help='video file name prefix: <name>-seed<seed>.mp4')
parser.add_argument('--video-dir', default='video')
parser.add_argument('--no-video', action='store_true', help='only print the scores, do not record')
parser.add_argument('--scale', type=float, default=0.5, help='resize factor of recorded frames (1.0 = 1000x800, needs a lot of RAM)')
parser.add_argument('--fps', type=int, default=50, help='CarRacing renders at 50 fps')
args = parser.parse_args()

# only the keys read by TD3BaseAgent / CarRacingTD3Agent are needed here
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
	"replay_buffer_capacity": 10,
	"logdir": '/tmp/td3_demo/',
	"update_freq": 2,
	"eval_interval": 10,
	"eval_episode": 10,
}

agent = CarRacingTD3Agent(config)
agent.load(args.model)
print(f"action limits used (brake_rate, gas_floor): {agent.limits()}")

# test_env uses render_mode="rgb_array"; env.env is the underlying gym env that can render frames.
# Frames are grabbed by hand (not with gym's RecordVideo): RecordVideo only ends a video when the
# underlying env terminates, but our wrapper ends episodes earlier (off-road rule), and RecordVideo
# throws away the frames of such unfinished episodes on the next reset().
env = agent.test_env


def grab_frame(frames):
	frame = env.env.render()
	if args.scale != 1.0:
		frame = cv2.resize(np.ascontiguousarray(frame), None, fx=args.scale, fy=args.scale,
		                   interpolation=cv2.INTER_AREA)
	frames.append(frame)


def save_video(frames, path):
	from moviepy.editor import ImageSequenceClip  # moviepy 1.0.3
	ImageSequenceClip(frames, fps=args.fps).write_videofile(path, logger=None)


if not args.no_video:
	os.makedirs(args.video_dir, exist_ok=True)

rewards = []
for ep in range(args.episodes):
	seed = args.seed + ep
	state, info = env.reset(seed=seed)
	frames = []
	if not args.no_video:
		grab_frame(frames)
	total_reward = 0
	length = 0
	while True:
		action = agent.decide_agent_actions(state)  # sigma = 0 -> deterministic
		state, reward, terminates, truncates, _ = env.step(action)
		total_reward += reward
		length += 1
		if not args.no_video:
			grab_frame(frames)
		if terminates or truncates:
			break
	print(f"Episode: {ep + 1}\tSeed: {seed}\tLength: {length}\tTotal reward: {total_reward:.2f}")
	if not args.no_video:
		path = os.path.join(args.video_dir, f"{args.name}-seed{seed}.mp4")
		save_video(frames, path)
		print(f"  video saved: {path}")
	rewards.append(total_reward)
env.close()
print(f"average score: {round(float(np.mean(rewards)), 2)}")
