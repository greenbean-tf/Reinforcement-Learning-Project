import argparse
from collections import deque
import itertools
import random
import time
import cv2

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class CarRacingEnvironment:
	# "custom" reward coefficients (heuristic values, not tuned)
	GRASS_COEF = 1.0    # penalty per step when the whole car window is grass (scaled by the grass ratio)
	JITTER_COEF = 0.3   # penalty per unit of steering change between two consecutive steps
	WINDOW_AREA = 24 * 20  # pixels in the crop used by check_car_position (obs[60:84, 40:60])
	# "custom_b" / "custom_bc" reward coefficients (heuristic values, not tuned)
	B_GRASS_COEF = 0.5       # penalty per step when the whole window is grass
	B_GRASS_THRESHOLD = 0.3  # only the grass ratio above this value is penalized

	def __init__(self, N_frame=4, test=False, reward_mode="original", speed_safe=None, speed_coef=0.5):
		self.test = test
		# "original" | "custom" (v1) | "custom_b" (grass only) | "custom_bc" (grass + speed-while-steering)
		self.reward_mode = reward_mode
		if reward_mode not in ("original", "custom", "custom_b", "custom_bc"):
			raise ValueError(f"unknown reward_mode: {reward_mode}")
		if reward_mode == "custom_bc" and not speed_safe:
			raise ValueError("custom_bc needs speed_safe > 0 (measure the car top speed first, see README)")
		self.speed_safe = speed_safe
		self.speed_coef = speed_coef
		self.prev_steer = None
		self.env = gym.make('CarRacing-v2', render_mode="rgb_array")
		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space
		self.ep_len = 0
		self.frames = deque(maxlen=N_frame)
	
	def check_car_position(self, obs):
		# cut the image to get the part where the car is
		part_image = obs[60:84, 40:60, :]

		road_color_lower = np.array([90, 90, 90], dtype=np.uint8)
		road_color_upper = np.array([120, 120, 120], dtype=np.uint8)
		grass_color_lower = np.array([90, 180, 90], dtype=np.uint8)
		grass_color_upper = np.array([120, 255, 120], dtype=np.uint8)
		road_mask = cv2.inRange(part_image, road_color_lower, road_color_upper)
		grass_mask = cv2.inRange(part_image, grass_color_lower, grass_color_upper)
		# count the number of pixels in the road and grass
		road_pixel_count = cv2.countNonZero(road_mask)
		grass_pixel_count = cv2.countNonZero(grass_mask)

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, part_image)

		return road_pixel_count, grass_pixel_count

	def step(self, action):
		obs, reward, terminates, truncates, info = self.env.step(action)
		original_reward = reward
		original_terminates = terminates
		self.ep_len += 1
		road_pixel_count, grass_pixel_count = self.check_car_position(obs)
		info["road_pixel_count"] = road_pixel_count
		info["grass_pixel_count"] = grass_pixel_count
		# car speed (same quantity as the speed indicator drawn by gym): norm of the hull's linear velocity
		vel = self.env.unwrapped.car.hull.linearVelocity
		speed = float(np.hypot(vel[0], vel[1]))
		info["speed"] = speed

		# off-track rule: end the episode with -100 once the car has left the road
		if road_pixel_count < 10:
			terminates = True
			reward = -100

		if self.reward_mode == "custom":
			# keep the original reward and the -100 / terminate rule above, and add two penalties
			# (only while the episode is still running, i.e. not when the -100 rule fired):
			# 1. grass penalty: the more grass under the car, the larger the penalty, so the agent
			#    is pushed back to the road before it fully leaves it (which ends the episode)
			# 2. steering-jitter penalty: discourage zig-zagging that leads to losing control
			if road_pixel_count >= 10:
				reward -= self.GRASS_COEF * grass_pixel_count / self.WINDOW_AREA
				steer = float(action[0])
				if self.prev_steer is not None:
					reward -= self.JITTER_COEF * abs(steer - self.prev_steer)
				self.prev_steer = steer
		elif self.reward_mode in ("custom_b", "custom_bc"):
			# B: original reward + a softer grass penalty that only counts the grass ratio above a threshold
			#    (no steering-jitter penalty, because sharp turns need large and fast steering)
			# C (only custom_bc): penalize speed above speed_safe while steering hard, so the agent learns
			#    to slow down before / during a sharp turn
			# Both are skipped when the -100 / terminate rule above has fired.
			if road_pixel_count >= 10:
				grass_ratio = grass_pixel_count / self.WINDOW_AREA
				reward -= self.B_GRASS_COEF * max(0.0, grass_ratio - self.B_GRASS_THRESHOLD)
				if self.reward_mode == "custom_bc":
					excess = max(0.0, speed - self.speed_safe) / self.speed_safe
					reward -= self.speed_coef * excess * abs(float(action[0]))

		# convert to grayscale
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# save image for debugging
		# filename = "images/image" + str(self.ep_len) + ".jpg"
		# cv2.imwrite(filename, obs)

		# frame stacking
		self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		if self.test:
			# enable this line to recover the original reward
			reward = original_reward
			# enable this line to recover the original terminates signal, disable this to accerlate evaluation
			# terminates = original_terminates

		return obs, reward, terminates, truncates, info
	
	def reset(self, seed=None):
		obs, info = self.env.reset(seed=seed)
		self.ep_len = 0
		self.prev_steer = None
		obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY) # 96x96

		# frame stacking
		for _ in range(self.frames.maxlen):
			self.frames.append(obs)
		obs = np.stack(self.frames, axis=0)

		return obs, info
	
	def render(self):
		self.env.render()
	
	def close(self):
		self.env.close()

if __name__ == '__main__':
	env = CarRacingEnvironment(test=True)
	obs, info = env.reset()
	done = False
	total_reward = 0
	total_length = 0
	t = 0
	while not done:
		t += 1
		action = env.action_space.sample()
		action[2] = 0.0
		obs, reward, terminates, truncates, info = env.step(action)
		print(f'{t}: road_pixel_count: {info["road_pixel_count"]}, grass_pixel_count: {info["grass_pixel_count"]}, reward: {reward}')
		total_reward += reward
		total_length += 1
		env.render()
		if terminates or truncates:
			done = True

	print("Total reward: ", total_reward)
	print("Total length: ", total_length)
	env.close()
