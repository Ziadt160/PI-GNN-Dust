"""Run a command once no other training job is using the machine.

Two heavy jobs on this 17 GB machine already killed each other once: both loaded
the 64,288 x 25 x 31 dataset at the same moment and both died with an allocation
failure, losing a download and an ablation arm. Everything queued behind another
run must therefore actually wait, not guess.

Guessing is what failed the second time. The first waiter polled a log file's
modification time, but the log was piped through `grep -v` to drop per-epoch
lines, so it sat silent for the ~50 minutes a long horizon takes. Silence read as
death and the waiter aborted a perfectly healthy queue.

So: poll the PROCESS TABLE, which cannot be filtered, buffered or misread.

Usage:
    python experiments/run_when_idle.py -- <command> [args...]
    python experiments/run_when_idle.py --match train_station_dust -- python foo.py
"""
import argparse
import subprocess
import sys
import time

import psutil

# A job is "ours" if its command line mentions one of these.
DEFAULT_MATCH = ("train_station_dust.py", "ablation_levels.py")
POLL_S = 60


def busy(patterns, exclude_pid):
    """Command lines of other live PYTHON jobs matching any pattern.

    The interpreter check is not cosmetic. Shell wrappers, grep pipelines and
    editor commands routinely carry a script's name in their own command line --
    a plain substring match found eight "jobs" when only two were real, and this
    waiter would then have blocked forever on a `grep` that had already exited.
    Only a process actually executing Python can be a competing trainer.
    """
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.info["pid"] in exclude_pid:
            continue
        if "python" not in (p.info["name"] or "").lower():
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if any(pat in cmd for pat in patterns):
            out.append((p.info["pid"], cmd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", action="append", default=None,
                    help="substring identifying a competing job "
                         f"(default: {', '.join(DEFAULT_MATCH)})")
    ap.add_argument("--timeout-min", type=float, default=720,
                    help="give up waiting after this long")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- then the command to run")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("nothing to run: pass the command after --")
    patterns = tuple(args.match or DEFAULT_MATCH)

    me = {psutil.Process().pid}
    deadline = time.time() + args.timeout_min * 60
    waited = 0.0
    while True:
        others = busy(patterns, me)
        if not others:
            break
        if time.time() > deadline:
            raise SystemExit(f"still busy after {args.timeout_min:.0f} min, "
                             f"giving up: {others[0][1][:90]}")
        if waited % 600 < POLL_S:          # a line every ~10 min, not every poll
            print(f"  waiting ({waited/60:.0f} min): {len(others)} job(s) live, "
                  f"e.g. pid {others[0][0]} {others[0][1][-70:]}", flush=True)
        time.sleep(POLL_S)
        waited += POLL_S

    print(f"machine idle after {waited/60:.0f} min -- starting:\n  "
          f"{' '.join(cmd)}\n", flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
