"""Run the pending training jobs strictly one at a time, and notice when one dies.

Why this exists rather than another shell chain
-----------------------------------------------
Three separate sequencing failures in one session, each from checking the wrong
signal for "is the previous job finished":

  * a waiter polled a LOG's mtime, but the log was piped through `grep -v` and
    went silent for the ~50 minutes a long horizon takes. Silence read as death
    and it aborted a healthy queue.
  * a waiter counted RESULT FILES, but results are now written after every
    horizon, so a partial file appears minutes in and looked like completion.
  * `TaskStop` killed a shell chain but not its grandchild, which sat holding
    5.5 GB. The next job started into too little memory and died. That orphan had
    ALREADY crashed -- traceback printed -- yet still held its allocation.

The reliable signal is **CPU time over an interval**. A working trainer burns CPU
continuously; a hung or dying one burns none while still holding gigabytes. That
distinction is invisible to every check above, and it is what STALL_MIN uses.

Jobs run sequentially in one process, so two trainers can never coexist -- the
collision that already cost this project a download and an ablation arm on a
17 GB machine.

Run:  python -u experiments/station_dust/run_queue.py
      python -u experiments/station_dust/run_queue.py --only smoke_physics auxcams_s0
      python -u experiments/station_dust/run_queue.py --list
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY = sys.executable
TRAIN = os.path.join("experiments", "station_dust", "train_station_dust.py")

# Every arm shares these, so a difference between arms is never an accident of
# the command line.
COMMON = ["--obs-input", "--upwind-obs", "--interior-only", "--daily",
          "--transport", "--hidden", "64", "--buffer-rings", "2"]

POLL_S = 30
STALL_MIN = 20          # no CPU for this long => hung or dying; kill it
IDLE_MATCH = ("train_station_dust.py", "ablation_levels.py")

# name -> extra flags. Order is deliberate: the cheap 2-epoch smoke test runs
# first because it is the only one exercising untested physics code, and a
# failure there should not cost six hours of queue behind it.
# Isolation of the physics encodings. physics_s0 turned all three on together
# and the model collapsed to a near-constant predictor -- flat 0.70/0.71/0.75
# validation at every horizon from epoch 2, against a baseline that learns
# horizon-specific 0.36/0.53/0.60, plus an epoch-1 loss of 40.86 at 24 h. Three
# changes went in at once, so the run cannot say WHICH broke.
#
# 3 epochs is enough: the signature is an epoch-1 explosion followed by a flat
# line, both visible immediately. --deposition cannot be tested alone because it
# reads the blh channel that --physics-features provides.
ISOLATION = [
    ("iso_control",      ["--epochs", "3", "--seed", "0", "--fresh"]),
    ("iso_features",     ["--epochs", "3", "--seed", "0", "--fresh",
                          "--physics-features"]),
    ("iso_deposition",   ["--epochs", "3", "--seed", "0", "--fresh",
                          "--physics-features", "--deposition"]),
    ("iso_conservative", ["--epochs", "3", "--seed", "0", "--fresh",
                          "--conservative"]),
]

# Re-measured after the normalisation fix (cbcc2e2). Every physics result so far
# was taken through a broken input pipeline -- hour-of-day normalised to ~1e6 --
# so none of them said anything about the physics itself. ctrl_fix is a matched
# control at the same seed and budget, since the fix also slightly changes the
# stats subsample for runs with no new channels.
#
# 5 epochs: the control separates its horizons by epoch 3 (0.37 / 0.53 / 0.59),
# so a healthy arm shows itself well before then; a collapsed one flatlines.
POSTFIX = [
    ("ctrl_fix", ["--epochs", "5", "--seed", "0", "--fresh"]),
    ("phys_fix", ["--epochs", "5", "--seed", "0", "--fresh",
                  "--physics-features", "--deposition", "--conservative"]),
]

JOBS = ISOLATION + POSTFIX + [
    # gated_s0 belongs here, not passed ad-hoc: --only filters THIS list, so a
    # name absent from it is silently skipped rather than run.
    ("smoke_physics", ["--epochs", "2", "--seed", "0", "--fresh",
                       "--physics-features", "--deposition", "--conservative"]),
    ("gated_s0",      ["--epochs", "60", "--seed", "0", "--fresh",
                       "--dust-gate"]),
    ("auxcams_s0",    ["--epochs", "60", "--seed", "0", "--fresh",
                       "--aux-cams", "0.05"]),
    # No --fresh: this arm has 3h and 24h complete and a 48h checkpoint at
    # epoch 15. The default architecture is unchanged by the new flags, so the
    # checkpoint still loads.
    ("abl_base_s1",   ["--epochs", "60", "--seed", "1"]),
    ("abl_ordinal_s0", ["--epochs", "60", "--seed", "0", "--fresh", "--ordinal"]),
    ("abl_ordinal_s1", ["--epochs", "60", "--seed", "1", "--fresh", "--ordinal"]),
]


def others_running(exclude):
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] in exclude:
            continue
        if "python" not in (p.info["name"] or "").lower():
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if any(m in cmd for m in IDLE_MATCH) and "run_queue.py" not in cmd:
            out.append((p.info["pid"], cmd))
    return out


def wait_for_idle(timeout_min=720):
    me = {psutil.Process().pid}
    deadline = time.time() + timeout_min * 60
    waited = 0
    while True:
        live = others_running(me)
        if not live:
            return
        if time.time() > deadline:
            raise SystemExit(f"still busy after {timeout_min} min: {live[0][1][:80]}")
        if waited % 600 == 0:
            print(f"  waiting for the machine ({waited//60} min): "
                  f"pid {live[0][0]} {live[0][1][-60:]}", flush=True)
        time.sleep(POLL_S)
        waited += POLL_S


def run_job(name, flags, log_dir):
    out = os.path.join("results", "ablation", f"{name}.json")
    log = os.path.join(log_dir, f"{name}.log")
    cmd = [PY, "-u", TRAIN] + flags + COMMON + ["--out", out]

    print(f"\n{'='*72}\n{name}\n  {' '.join(flags)}\n  log -> {log}\n{'='*72}",
          flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=fh,
                                stderr=subprocess.STDOUT)
        ps = psutil.Process(proc.pid)
        last_cpu, last_move = 0.0, time.time()
        while proc.poll() is None:
            time.sleep(POLL_S)
            try:
                cpu = ps.cpu_times().user + ps.cpu_times().system
            except psutil.Error:
                break
            if cpu - last_cpu > 1.0:
                last_cpu, last_move = cpu, time.time()
            elif time.time() - last_move > STALL_MIN * 60:
                # A crashed-but-unreaped process keeps its memory and blocks the
                # rest of the queue. terminate() did not shift one of these; kill
                # is what worked.
                print(f"  STALLED: no CPU for {STALL_MIN} min, killing "
                      f"(rss {ps.memory_info().rss/1e9:.1f} GB)", flush=True)
                try:
                    ps.kill()
                except psutil.Error:
                    pass
                break
        rc = proc.wait()

    mins = (time.time() - t0) / 60
    done = []
    if os.path.exists(out):
        try:
            done = json.load(open(out, encoding="utf-8")).get("complete") or []
        except (json.JSONDecodeError, OSError):
            pass
    print(f"  {name}: exit {rc} after {mins:.1f} min | horizons {done or 'none'}",
          flush=True)
    return {"job": name, "exit": rc, "minutes": round(mins, 1), "complete": done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run just these jobs, by name")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, f in JOBS:
            print(f"  {n:<18} {' '.join(f)}")
        return

    jobs = [(n, f) for n, f in JOBS if not args.only or n in args.only]
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "ablation"), exist_ok=True)

    print(f"queue: {', '.join(n for n, _ in jobs)}", flush=True)
    wait_for_idle()

    summary = []
    for name, flags in jobs:
        summary.append(run_job(name, flags, log_dir))
        with open(os.path.join("results", "ablation", "queue_status.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    print(f"\n{'='*72}\nqueue finished")
    for s in summary:
        flag = "ok " if s["exit"] == 0 else "FAIL"
        print(f"  [{flag}] {s['job']:<18} {s['minutes']:>6.1f} min  "
              f"horizons {s['complete'] or 'none'}")


if __name__ == "__main__":
    main()
