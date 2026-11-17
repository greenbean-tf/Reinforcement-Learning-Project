"""Read TensorBoard event files under --logroot and draw one comparison PNG per ablation.

Usage (from the src directory):
	python plot_results.py                       # reads log/CarRacing, writes log/CarRacing/plots
	python plot_results.py --logroot log/CarRacing --smooth 0.9

Each experiment must live in its own folder, e.g. log/CarRacing/baseline, log/CarRacing/single_q ...
Missing folders are skipped, so it is fine to run this before every experiment has finished.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TRAIN_TAG = "Train/Episode Reward"
EVAL_TAG = "Evaluate/Episode Reward"

# ablation name -> list of (legend label, folder name under logroot)
GROUPS = {
	"0_baseline": [("baseline", "baseline")],
	"1_twin_q": [("twin Q (baseline)", "baseline"), ("single Q", "single_q")],
	"2_target_smoothing": [("smoothing on (baseline)", "baseline"), ("smoothing off", "no_smoothing")],
	"3_delayed_update": [("update_freq=1", "delay1"), ("update_freq=2 (baseline)", "baseline"), ("update_freq=4", "delay4")],
	"4_action_noise": [("sigma_max=0.03", "sigma0.03"), ("sigma_max=0.1 (baseline)", "baseline"), ("sigma_max=0.3", "sigma0.3")],
	# strong setting (limits relaxed + noise decay, smoothing on): the three rewards (200k steps) and the 150k baseline
	# final run: T_bc continued to 1M steps with a larger GPU buffer (F_bc starts from the last T_bc state at ~390k steps)
	"8_final_run": [("T_bc (400k)", "T_bc"), ("F_bc (continued, 1M)", "F_bc")],
	# next generation (400k steps): bigger GPU buffer, rarer evaluation, stuck rule; T_b_p300 = stuck penalty -300
	"7_next_generation": [("baseline (150k)", "baseline"), ("S_b (200k)", "S_b"), ("T_b", "T_b"),
	                      ("T_b, stuck penalty -300", "T_b_p300"), ("T_orig", "T_orig"), ("T_bc", "T_bc")],
	"6_strong_setting_rewards": [("baseline (150k)", "baseline"), ("strong, original reward", "S_orig"),
	                             ("strong, custom B", "S_b"), ("strong, custom B+C", "S_bc")],
	"5_reward_function": [("original reward (baseline)", "baseline"), ("custom reward v1", "custom_reward"),
	                      ("custom B", "custom_b"), ("custom B+C", "custom_bc")],
}


def load_scalars(run_dir, tag):
	"""Return (steps, values) of a scalar tag; empty lists if the tag is missing."""
	acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = keep every point
	acc.Reload()
	if tag not in acc.Tags().get("scalars", []):
		return [], []
	events = acc.Scalars(tag)
	return [e.step for e in events], [e.value for e in events]


def ema(values, weight):
	"""Exponential moving average, same idea as TensorBoard's smoothing slider."""
	out, last = [], None
	for v in values:
		last = v if last is None else last * weight + (1 - weight) * v
		out.append(last)
	return out


def plot_group(name, runs, logroot, outdir, smooth):
	# a comparison needs at least 2 existing runs (a lone baseline is only drawn for group 0)
	available = [r for r in runs if os.path.isdir(os.path.join(logroot, r[1]))]
	if len(available) < min(2, len(runs)):
		print(f"[{name}] skipped: found {len(available)} of {len(runs)} runs")
		return None
	fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
	drawn = 0
	for label, folder in runs:
		run_dir = os.path.join(logroot, folder)
		if not os.path.isdir(run_dir):
			print(f"  [skip] {run_dir} not found")
			continue
		for ax, tag in zip(axes, (TRAIN_TAG, EVAL_TAG)):
			steps, values = load_scalars(run_dir, tag)
			if not steps:
				continue
			if tag == TRAIN_TAG:
				line, = ax.plot(steps, ema(values, smooth), label=label)
				ax.plot(steps, values, color=line.get_color(), alpha=0.2)  # raw curve, faint
			else:
				# evaluation points are sparse (one per eval_interval episodes): no smoothing
				ax.plot(steps, values, marker="o", markersize=3, label=label)
			drawn += 1
	if drawn == 0:
		plt.close(fig)
		print(f"[{name}] nothing to draw")
		return None
	for ax, title in zip(axes, ("Train episode reward", "Evaluation episode reward (avg)")):
		ax.set_title(title)
		ax.set_xlabel("environment steps")
		ax.set_ylabel("reward")
		ax.grid(alpha=0.3)
		ax.legend()
	fig.suptitle(name.split("_", 1)[1].replace("_", " "))
	fig.tight_layout()
	path = os.path.join(outdir, f"{name}.png")
	fig.savefig(path, dpi=150)
	plt.close(fig)
	print(f"[{name}] saved {path}")
	return path


def summarize(logroot, outdir):
	"""Print and save best / last evaluation reward of every run folder."""
	rows = []
	for folder in sorted(os.listdir(logroot)):
		run_dir = os.path.join(logroot, folder)
		if not os.path.isdir(run_dir) or folder == "plots":
			continue
		steps, values = load_scalars(run_dir, EVAL_TAG)
		if not steps:
			continue
		rows.append((folder, len(steps), max(values), values[-1], steps[-1]))
	if not rows:
		return
	print("\nrun                    evals   best_eval   last_eval   last_step")
	for r in rows:
		print(f"{r[0]:<22} {r[1]:>5}   {r[2]:>9.1f}   {r[3]:>9.1f}   {r[4]:>9}")
	with open(os.path.join(outdir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["run", "num_evals", "best_eval_reward", "last_eval_reward", "last_step"])
		w.writerows(rows)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--logroot", default="log/CarRacing", help="folder that holds one sub-folder per experiment")
	parser.add_argument("--out", default=None, help="output folder for PNGs (default: <logroot>/plots)")
	parser.add_argument("--smooth", type=float, default=0.9, help="EMA smoothing weight in [0, 1)")
	args = parser.parse_args()

	outdir = args.out or os.path.join(args.logroot, "plots")
	os.makedirs(outdir, exist_ok=True)
	for name, runs in GROUPS.items():
		plot_group(name, runs, args.logroot, outdir, args.smooth)
	summarize(args.logroot, outdir)
