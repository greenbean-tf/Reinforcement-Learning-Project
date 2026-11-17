"""Pick the best checkpoint: re-evaluate many checkpoints on the SAME fresh seeds and rank them.

The evaluation score stored in a checkpoint's file name (model_<step>_<score>.pth) is the average of only 10 random
tracks and is a noisy, upward-biased pick (it was chosen as the best of many noisy evaluations). This script
re-tests the most promising checkpoints of each run on a common set of tracks (seeds), so that they can be compared
fairly, and writes one CSV row per checkpoint. Re-running it continues where it stopped: checkpoints that already
have a row for the same seeds are skipped.

Examples (run from the src directory):
	# stage 1: top-3 and last-2 checkpoints of each run on 20 fresh tracks
	python select_model.py --runs T_orig,T_b_p300,T_bc,baseline --top 3 --last 2 --seeds 100-119

	# stage 2: the few best candidates on a separate set of held-out tracks
	python select_model.py --paths log/CarRacing/F_bc/model_888858_889.pth,log/CarRacing/T_bc/model_293574_858.pth --seeds 200-219 --out log/CarRacing/select_holdout.csv

Parallel evaluation: add --workers N (default 1) to evaluate N checkpoints at the same time in N processes. The games
are CPU bound (Box2D), so use at most as many workers as there are CPU cores (Colab: 2); every worker needs its own
CUDA context (a few hundred MB of GPU memory), so do not start workers while a training run holds almost all of the GPU.

Seeds: "100-119", "3,17,42" or a mix such as "100-104,200".
"""
import argparse
import csv
import atexit
import concurrent.futures
import glob
import importlib
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np

FIELDS = ["run", "step", "train_eval_score", "seeds", "n_games", "mean", "std", "min", "max", "n_short",
          "brake_rate", "gas_floor", "path", "rewards", "lengths"]
SHORT_EPISODE = 400  # games that end before this many steps count as "short" (the car left the road early)


def parse_seeds(text):
	seeds = []
	for part in str(text).split(","):
		part = part.strip()
		if not part:
			continue
		m = re.fullmatch(r"(\d+)-(\d+)", part)
		if m:
			seeds.extend(range(int(m.group(1)), int(m.group(2)) + 1))
		else:
			seeds.append(int(part))
	assert seeds, "no seeds given"
	return seeds


def find_candidates(logroot, runs, top, last):
	"""For every run: the `top` checkpoints with the highest in-training score and the `last` most recent ones."""
	pattern = re.compile(r"model_(\d+)_(-?\d+)\.pth$")
	found = []
	for run in runs:
		files = []
		for path in glob.glob(os.path.join(logroot, run, "model_*_*.pth")):
			m = pattern.search(os.path.basename(path))
			if m:
				files.append({"run": run, "step": int(m.group(1)), "score": int(m.group(2)), "path": path})
		if not files:
			print(f"[warn] no checkpoints found for run '{run}' under {logroot}")
			continue
		by_score = sorted(files, key=lambda d: d["score"], reverse=True)[:top]
		by_step = sorted(files, key=lambda d: d["step"])[-last:] if last > 0 else []
		chosen = by_score + by_step
		seen = set()
		for d in chosen:
			if d["path"] not in seen:
				seen.add(d["path"])
				found.append(d)
	return found


def build_agent(tmpdir):
	"""An agent used only for evaluation (the replay buffer is allocated lazily, so it costs nothing)."""
	factory = os.environ.get("SELECT_MODEL_AGENT_FACTORY")  # "module:function", for tests only
	if factory:
		module, attr = factory.split(":")
		return getattr(importlib.import_module(module), attr)(tmpdir)
	from td3_agent_CarRacing import CarRacingTD3Agent
	config = {
		"gpu": True, "training_steps": 1e8, "gamma": 0.99, "tau": 0.005, "batch_size": 32, "warmup_steps": 1000,
		"total_episode": 100000, "lra": 4.5e-5, "lrc": 4.5e-5, "replay_buffer_capacity": 10, "logdir": tmpdir,
		"update_freq": 2, "eval_interval": 10, "eval_episode": 10,
	}
	return CarRacingTD3Agent(config)


def evaluate_checkpoint(agent, path, seeds):
	"""Play one game per seed (no exploration noise) with the original environment reward."""
	agent.fixed_limits = None  # a legacy checkpoint has no saved limits -> the original limits
	agent.load(path)           # a new checkpoint restores the brake / gas limits it was trained with
	rewards, lengths = [], []
	for seed in seeds:
		state, _ = agent.test_env.reset(seed=seed)
		total, steps = 0.0, 0
		while True:
			action = agent.decide_agent_actions(state)
			state, reward, terminates, truncates, _ = agent.test_env.step(action)
			total += reward
			steps += 1
			if terminates or truncates:
				break
		rewards.append(total)
		lengths.append(steps)
	return rewards, lengths


def make_row(c, seeds, seeds_text, rewards, lengths, brake, gas):
	return {
		"run": c["run"], "step": c["step"], "train_eval_score": c["score"], "seeds": seeds_text,
		"n_games": len(seeds), "mean": f"{np.mean(rewards):.2f}",
		"std": f"{np.std(rewards, ddof=1) if len(seeds) > 1 else 0.0:.2f}",
		"min": f"{np.min(rewards):.2f}", "max": f"{np.max(rewards):.2f}",
		"n_short": int(sum(1 for n in lengths if n < SHORT_EPISODE)),
		"brake_rate": f"{brake:.4f}", "gas_floor": f"{gas:.4f}", "path": c["path"],
		"rewards": ";".join(f"{r:.1f}" for r in rewards), "lengths": ";".join(str(n) for n in lengths),
	}


_AGENT = None  # one evaluation agent per process (created on first use)


def _worker_agent():
	global _AGENT
	if _AGENT is None:
		tmpdir = tempfile.mkdtemp(prefix="td3_select_")
		atexit.register(shutil.rmtree, tmpdir, True)
		_AGENT = build_agent(tmpdir)
	return _AGENT


def _eval_task(c, seeds):
	"""Evaluate one checkpoint (runs in the main process or in a worker process)."""
	t0 = time.time()
	agent = _worker_agent()
	rewards, lengths = evaluate_checkpoint(agent, c["path"], seeds)
	brake, gas = agent.limits() if hasattr(agent, "limits") else (float("nan"), float("nan"))
	return c, rewards, lengths, brake, gas, time.time() - t0


def load_rows(csv_path):
	if not os.path.exists(csv_path):
		return []
	with open(csv_path, newline="", encoding="utf-8") as f:
		return list(csv.DictReader(f))


def append_row(csv_path, row):
	new = not os.path.exists(csv_path)
	with open(csv_path, "a", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=FIELDS)
		if new:
			w.writeheader()
		w.writerow(row)


def print_ranking(rows):
	rows = sorted(rows, key=lambda r: float(r["mean"]), reverse=True)
	print("\nrank  run             step      train-eval  mean    std     SE     min     short  checkpoint")
	for i, r in enumerate(rows, 1):
		n = int(r["n_games"])
		se = float(r["std"]) / np.sqrt(n)
		print(f"{i:>4}  {r['run']:<14} {int(r['step']):>8}  {r['train_eval_score']:>10}  {float(r['mean']):>6.1f}  "
		      f"{float(r['std']):>6.1f}  {se:>5.1f}  {float(r['min']):>6.1f}  {r['n_short']:>5}  "
		      f"{os.path.basename(r['path'])}")
	print("\n'short' = games that ended before %d steps (usually a crash off the road). "
	      "SE = std / sqrt(games): differences smaller than ~2 SE are noise." % SHORT_EPISODE)


def main(argv=None):
	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--logroot", default="log/CarRacing", help="folder with one sub-folder per run")
	p.add_argument("--runs", default="", help="comma separated run names (sub-folders of --logroot)")
	p.add_argument("--top", type=int, default=3, help="per run: number of checkpoints with the best in-training score")
	p.add_argument("--last", type=int, default=2, help="per run: number of most recent checkpoints")
	p.add_argument("--paths", default="", help="comma separated checkpoint paths to evaluate in addition")
	p.add_argument("--seeds", default="100-119", help='seeds of the test tracks, e.g. "100-119" or "3,17,42"')
	p.add_argument("--out", default=None, help="CSV file (default: <logroot>/select_results.csv)")
	p.add_argument("--workers", type=int, default=1, help="evaluate this many checkpoints in parallel (processes)")
	args = p.parse_args(argv)

	seeds = parse_seeds(args.seeds)
	seeds_text = args.seeds.replace(" ", "")
	out = args.out or os.path.join(args.logroot, "select_results.csv")
	runs = [r.strip() for r in args.runs.split(",") if r.strip()]
	cands = find_candidates(args.logroot, runs, args.top, args.last) if runs else []
	for path in [x.strip() for x in args.paths.split(",") if x.strip()]:
		m = re.search(r"model_(\d+)_(-?\d+)\.pth$", os.path.basename(path))
		cands.append({"run": os.path.basename(os.path.dirname(os.path.abspath(path))) or "manual",
		              "step": int(m.group(1)) if m else 0, "score": int(m.group(2)) if m else 0, "path": path})
	if not cands:
		sys.exit("no checkpoints to evaluate (use --runs and/or --paths)")

	done = {r["path"] for r in load_rows(out) if r["seeds"] == seeds_text}
	todo = [c for c in cands if c["path"] not in done]
	print(f"{len(cands)} candidate checkpoints, {len(cands) - len(todo)} already evaluated on seeds {seeds_text}, "
	      f"{len(todo)} to do ({len(seeds)} games each)")

	def record(i, c, rewards, lengths, brake, gas, secs):
		row = make_row(c, seeds, seeds_text, rewards, lengths, brake, gas)
		append_row(out, row)  # written by the main process only, one row as soon as a checkpoint is done
		print(f"[{i}/{len(todo)}] {c['run']} step {c['step']} (train-eval {c['score']}): mean {row['mean']} "
		      f"± {row['std']} over {len(seeds)} games, {row['n_short']} short, {secs:.0f}s", flush=True)

	workers = max(1, min(args.workers, len(todo)))
	if todo and workers > 1:
		print(f"evaluating with {workers} worker processes")
		ctx = multiprocessing.get_context("spawn")  # CUDA cannot be used in a forked process
		with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
			futures = [pool.submit(_eval_task, c, seeds) for c in todo]
			for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
				record(i, *fut.result())
	else:
		for i, c in enumerate(todo, 1):
			record(i, *_eval_task(c, seeds))

	rows = [r for r in load_rows(out) if r["seeds"] == seeds_text]
	print_ranking(rows)
	print(f"\nresults saved to {out}")
	return rows


if __name__ == "__main__":
	main()
