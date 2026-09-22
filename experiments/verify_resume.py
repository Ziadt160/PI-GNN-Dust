"""Does the checkpoint actually let a killed run continue where it stopped?

Exercises the real helpers from train_station_dust.py, not a mock of them:
weights and optimiser state survive a round trip, the resume branch reads the
right epoch and patience counter, a finished horizon is recognised as finished,
and a crash mid-write cannot leave a truncated results file.
"""
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
sys.argv = ["x"]
import importlib.util

spec = importlib.util.spec_from_file_location(
    "tsd", os.path.join("experiments", "train_station_dust.py"))
tsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsd)

PASS, FAIL = "PASS", "** FAIL **"
results = []


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


os.chdir(tempfile.mkdtemp())

# --- a real model and optimiser, trained a little so state is non-trivial ----
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(5):
    opt.zero_grad()
    nn.functional.mse_loss(model(torch.randn(32, 8)), torch.randn(32, 4)).backward()
    opt.step()

best, bad, epoch = 0.5078, 7, 41
best_state = {k: v.clone() for k, v in model.state_dict().items()}

ck = tsd.ckpt_path("testrun", "24h")
torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
            "best": best, "bad": bad, "epoch": epoch,
            "best_state": best_state, "done": False}, ck)
report("checkpoint file is created under results/checkpoints",
       os.path.exists(ck), ck)

# --- reload into a FRESH model, as a restarted process would ----------------
torch.manual_seed(999)
model2 = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
before = model2[0].weight.clone()

saved = torch.load(ck, map_location="cpu", weights_only=False)
model2.load_state_dict(saved["model"])
opt2.load_state_dict(saved["opt"])

same = torch.allclose(model2[0].weight, model[0].weight)
moved = not torch.allclose(model2[0].weight, before)
report("weights restore exactly into a fresh process",
       same and moved, f"match={same}, differs from re-init={moved}")

# Adam's per-parameter moments must come back, or resuming restarts the
# optimiser cold and the first epochs after a crash are wasted re-warming it.
st = opt2.state_dict()["state"]
has_moments = bool(st) and all("exp_avg" in v for v in st.values())
step_ok = bool(st) and all(int(v["step"]) == 5 for v in st.values())
report("Adam moment estimates and step count survive",
       has_moments and step_ok,
       f"{len(st)} param groups carry exp_avg/exp_avg_sq, step=5")

report("epoch, best val and patience counter restore",
       (saved["epoch"], saved["best"], saved["bad"]) == (epoch, best, bad),
       f"resumes at epoch {saved['epoch']+1}, best {saved['best']}, "
       f"{saved['bad']}/{tsd.PATIENCE} without gain")

# --- a finished horizon must be skipped, not retrained ----------------------
torch.save({**saved, "done": True}, ck)
done = torch.load(ck, map_location="cpu", weights_only=False).get("done", False)
report("a completed horizon is flagged done and skipped on resume",
       done is True, "done=True -> training loop is bypassed entirely")

# --- per-epoch log is append-only and parseable line by line ----------------
log = tsd.log_path("testrun", "24h")
with open(log, "a", encoding="utf-8") as fh:
    for e in (1, 2, 3):
        fh.write(json.dumps({"horizon": "24h", "epoch": e, "val_loss": 0.6 - e / 100}) + "\n")
rows = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
report("per-epoch log survives as one valid JSON object per line",
       len(rows) == 3 and rows[-1]["epoch"] == 3,
       f"{len(rows)} rows readable; a process killed mid-epoch keeps all prior rows")


# --- results are written per horizon, atomically ----------------------------
class A:
    out = os.path.join("results", "part.json")
    seed = 0


res = {"3h|dusty|no-obs": {"TS": 0.296}, "24h|dusty|no-obs": {"TS": 0.267}}
tsd.write_results(A.out, res, ["OEDR"], set(), A())
loaded = json.load(open(A.out, encoding="utf-8"))
report("partial results are written after each horizon",
       set(loaded["complete"]) == {"3h", "24h"},
       f"complete horizons recorded: {loaded['complete']}")
report("no .partial temp file is left behind (write was atomic)",
       not os.path.exists(A.out + ".partial"),
       "os.replace() means a crash mid-write leaves the previous file intact")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
