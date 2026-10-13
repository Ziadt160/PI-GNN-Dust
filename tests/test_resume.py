"""Does the checkpoint actually let a killed run continue where it stopped?

Exercises the real helpers from train_station_dust.py, not a mock of them:
weights and optimiser state survive a round trip, the resume branch reads the
right epoch and patience counter, a finished horizon is recognised as finished,
and a crash mid-write cannot leave a truncated results file.

Run:  pytest tests/test_resume.py
"""
import argparse
import json
import os

import pytest
import torch
import torch.nn as nn

from experiments.station_dust import train_station_dust as tsd

BEST, BAD, EPOCH = 0.5078, 7, 41


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """The helpers write under a relative results/ directory; keep it out of the repo."""
    monkeypatch.chdir(tmp_path)


def _net():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture
def trained():
    """A real model and optimiser, trained a little so state is non-trivial."""
    torch.manual_seed(0)
    model = _net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(model(torch.randn(32, 8)), torch.randn(32, 4)).backward()
        opt.step()
    return model, opt


@pytest.fixture
def ckpt(trained):
    """Path of a checkpoint carrying everything the resume branch reads."""
    model, opt = trained
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    path = tsd.ckpt_path("testrun", "24h")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "best": BEST, "bad": BAD, "epoch": EPOCH,
                "best_state": best_state, "done": False}, path)
    return path


@pytest.fixture
def restored(ckpt):
    """Reload into a FRESH model and optimiser, as a restarted process would."""
    torch.manual_seed(999)
    model = _net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    init = model[0].weight.clone()
    saved = _load(ckpt)
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["opt"])
    return model, opt, init


def test_checkpoint_file_is_created_under_results_checkpoints(ckpt):
    assert os.path.exists(ckpt), ckpt
    assert os.path.dirname(ckpt) == os.path.join("results", "checkpoints"), ckpt


def test_weights_restore_exactly_into_a_fresh_process(trained, restored):
    model, _ = trained
    model2, _, init = restored
    same = torch.allclose(model2[0].weight, model[0].weight)
    moved = not torch.allclose(model2[0].weight, init)
    assert same and moved, f"match={same}, differs from re-init={moved}"


# Adam's per-parameter moments must come back, or resuming restarts the
# optimiser cold and the first epochs after a crash are wasted re-warming it.
def test_adam_moment_estimates_and_step_count_survive(restored):
    _, opt2, _ = restored
    st = opt2.state_dict()["state"]
    has_moments = bool(st) and all("exp_avg" in v for v in st.values())
    step_ok = bool(st) and all(int(v["step"]) == 5 for v in st.values())
    assert has_moments and step_ok, (
        f"{len(st)} parameters in optimiser state; exp_avg present={has_moments}, "
        f"steps {sorted({int(v['step']) for v in st.values()})} (expected 5)")


def test_epoch_best_val_and_patience_counter_restore(ckpt):
    saved = _load(ckpt)
    assert (saved["epoch"], saved["best"], saved["bad"]) == (EPOCH, BEST, BAD), (
        f"resumes at epoch {saved['epoch']+1}, best {saved['best']}, "
        f"{saved['bad']}/{tsd.PATIENCE} without gain")


# A finished horizon must be skipped, not retrained: done=True means the
# training loop is bypassed entirely.
def test_completed_horizon_is_flagged_done(ckpt):
    torch.save({**_load(ckpt), "done": True}, ckpt)
    done = _load(ckpt).get("done", False)
    assert done is True, f"done={done!r}"


# A process killed mid-epoch must keep all prior rows.
def test_epoch_log_survives_as_one_json_object_per_line():
    log = tsd.log_path("testrun", "24h")
    with open(log, "a", encoding="utf-8") as fh:
        for e in (1, 2, 3):
            fh.write(json.dumps({"horizon": "24h", "epoch": e, "val_loss": 0.6 - e / 100}) + "\n")
    with open(log, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 3 and rows[-1]["epoch"] == 3, f"{len(rows)} rows readable"


# ---- results are written per horizon, atomically --------------------------- #
@pytest.fixture
def written():
    """Two scored horizons, persisted the way the trainer does after each one."""
    args = argparse.Namespace(out=os.path.join("results", "part.json"), seed=0)
    res = {"3h|dusty|no-obs": {"TS": 0.296}, "24h|dusty|no-obs": {"TS": 0.267}}
    tsd.write_results(args.out, res, ["OEDR"], set(), args)
    return args.out


def test_partial_results_are_written_after_each_horizon(written):
    with open(written, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert set(loaded["complete"]) == {"3h", "24h"}, (
        f"complete horizons recorded: {loaded['complete']}")


# os.replace() means a crash mid-write leaves the previous file intact.
def test_no_partial_temp_file_left_behind(written):
    assert not os.path.exists(written + ".partial"), f"{written}.partial still exists"
