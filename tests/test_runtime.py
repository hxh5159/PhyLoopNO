"""Synthetic checks for configuration, data RNG and actual training archives."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from linearno_loop.v5.config import resolve_config, validate_config
from linearno_loop.v5.contracts import ARCHITECTURE


def small_config(task="elasticity", **options):
    selected = dict(architecture=ARCHITECTURE, expert_count=2, expert_width=8)
    selected.update(options)
    return resolve_config(task, options=selected, profile_overrides={
        "model.hidden": 8, "model.heads": 2, "model.linearno_rank": 4,
        "training.epochs": 3,
    })


@pytest.mark.parametrize("task,count", [
    ("airfoil", 4), ("pipe", 3), ("plasticity", 4),
    ("darcy", 1), ("ns", 4), ("elasticity", 4),
])
def test_paper_defaults_and_depth_schedules(task, count):
    cfg = resolve_config(task, options={"architecture": ARCHITECTURE})
    assert cfg["expert_count"] == count
    assert (cfg["prefix_blocks"], cfg["recurrent_core_blocks"],
            cfg["loop_repeats"], cfg["suffix_blocks"]) == (1, 3, 2, 1)
    assert validate_config(cfg) == cfg
    for depth, expected in [(8, (1, 3, 2, 1)), (16, (2, 6, 2, 2)),
                            (24, (2, 10, 2, 2)), (32, (2, 14, 2, 2)),
                            (40, (2, 18, 2, 2))]:
        resolved = resolve_config(task, options={"architecture": ARCHITECTURE,
                                               "executed_depth": depth})
        assert tuple(resolved[key] for key in ("prefix_blocks", "recurrent_core_blocks",
                                               "loop_repeats", "suffix_blocks")) == expected
        assert resolved["loop_spec"]["executed_depth"] == depth


@pytest.mark.parametrize("task", ["airfrans", "car"])
def test_industrial_expert_count_is_explicit(task):
    with pytest.raises(ValueError, match="expert_count"):
        resolve_config(task, options={"architecture": ARCHITECTURE})
    assert resolve_config(task, options={"architecture": ARCHITECTURE,
                                        "expert_count": 3})["expert_count"] == 3


def test_generator_collate_owns_time_permutation_rng():
    from cdlno.linearno_loop.standard_entry import GeneratorCollate
    generator = torch.Generator().manual_seed(17)
    def collate(batch):
        return torch.stack([torch.randperm(20) for _ in batch])
    wrapped = GeneratorCollate(collate, generator)
    public = torch.random.get_rng_state().clone()
    state = generator.get_state().clone()
    first = wrapped([0, 1])
    torch.testing.assert_close(torch.random.get_rng_state(), public, rtol=0, atol=0)
    assert not torch.equal(state, generator.get_state())
    generator.set_state(state)
    torch.rand(100)  # Other model work must not change the time queries.
    torch.testing.assert_close(wrapped([0, 1]), first, rtol=0, atol=0)


class Recorder:
    def __init__(self, directory):
        self.result_dir = directory
    def attach_model(self, model):
        pass
    def update_protocol(self, protocol):
        pass


def test_standard_resume_restores_actual_updates_and_loader_rng(tmp_path, monkeypatch):
    from cdlno.linearno_loop.standard_entry import LoopStandardRun
    from cdlno.linearno_loop.v5.construction import build_from_config
    from cdlno.linearno_loop.v5.checkpoint import inspect_checkpoint, read_pair
    import cdlno.experiment
    import cdlno.linearno_loop.versioning as runtime

    monkeypatch.setattr(cdlno.experiment, "session", lambda args: Recorder(tmp_path))
    monkeypatch.setattr(runtime, "provenance", lambda *a, **k: {"source_sha256": "a" * 64})
    cfg = small_config()
    positions = torch.linspace(0, 1, 24).reshape(2, 6, 2)
    target = positions.square().sum(-1, keepdim=True)
    dataset = TensorDataset(positions, target)

    def setup(resume=False, checkpoint=None):
        args = SimpleNamespace(linearno_run_dir=tmp_path, linearno_task="elasticity",
            _linearno_loop_config=cfg, _linearno_config=cfg["profile_spec"],
            _linearno_normalizers={}, epochs=3, eval=0, resume=resume,
            _linearno_data=dict(checksums={"synthetic": "b" * 64},
                split="SYNTHETIC two samples", sampling="all six points", scope="synthetic"))
        if checkpoint is not None:
            args._linearno_checkpoint = checkpoint
        model = build_from_config(cfg)
        run = LoopStandardRun(args, model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        loaders = (DataLoader(dataset, batch_size=1, shuffle=True),
                   DataLoader(dataset, batch_size=1))
        run.prepare(optimizer, scheduler, *loaders)
        return run, model, optimizer, scheduler, loaders[0]

    def epoch(run, model, optimizer, scheduler, loader):
        losses = []
        for x, y in loader:
            optimizer.zero_grad()
            loss = (model(x, None) - y).square().mean()
            loss.backward(); optimizer.step()
            losses.append(loss.detach().clone())
        scheduler.step()
        run.complete_epoch(run.completed_epoch + 1)
        return torch.stack(losses)

    run, model, optimizer, scheduler, loader = setup()
    epoch(run, model, optimizer, scheduler, loader)
    manifest = run.save(model)
    metadata, path = inspect_checkpoint(tmp_path, "latest", expected=cfg)
    assert metadata["resume_state"]["epoch"] == 1
    assert path == manifest
    expected_losses = epoch(run, model, optimizer, scheduler, loader)
    expected_state = copy.deepcopy(model.state_dict())
    expected_rng = torch.random.get_rng_state().clone()
    resumed, model2, optimizer2, scheduler2, loader2 = setup(True, manifest)
    assert resumed.start_epoch == 1
    actual_losses = epoch(resumed, model2, optimizer2, scheduler2, loader2)
    torch.testing.assert_close(actual_losses, expected_losses, rtol=0, atol=0)
    for key, value in expected_state.items():
        torch.testing.assert_close(model2.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert scheduler2.state_dict() == scheduler.state_dict()
    resumed.save(model2)
    assert read_pair(tmp_path / "checkpoints/epoch_0002.json", expected=cfg)[0]["resume_state"]["epoch"] == 2


def test_standalone_car_training_loop_uses_native_objective(tmp_path, monkeypatch):
    from torch_geometric.data import Data
    from cdlno.linearno_loop.car_entry import CarRun
    from cdlno.linearno_loop.v5.construction import build_from_config
    from cdlno.linearno_loop.v5.checkpoint import inspect_checkpoint
    import cdlno.linearno_loop.car_entry as entry
    import numpy as np

    monkeypatch.setattr(entry, "provenance", lambda *a, **k: {"source_sha256": "c" * 64})
    cfg = small_config("car")
    args = SimpleNamespace(_linearno_config=cfg["profile_spec"],
        _linearno_loop_config=cfg, linearno_run_dir=tmp_path,
        nb_epochs=3, resume=False, eval=0, device="cpu", seed=0, linearno_task="car")
    coef = (np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32),
            np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    data_spec = dict(checksums={"synthetic": "d" * 64}, split="SYNTHETIC graphs",
                     sampling="all six nodes", ntrain=2, ntest=1)
    run = CarRun(args, data_spec, coef)
    model = build_from_config(cfg)
    x = torch.arange(42, dtype=torch.float32).reshape(6, 7) / 42
    y = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24
    surf = torch.tensor([True, True, True, False, False, False])
    datasets = [(Data(x=x.clone(), y=y.clone(), surf=surf.clone()),
                 Data(x=torch.ones(2, 3))) for _ in range(2)]
    output = model(datasets[0])
    total, pressure, velocity = run.objective(output, datasets[0][0])
    torch.testing.assert_close(total, velocity + .5 * pressure)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run.train("cpu", datasets, datasets[:1], model,
        dict(lr=1e-3, batch_size=1, nb_epochs=3), str(tmp_path), .5, 1, coef, None)
    saved, _ = inspect_checkpoint(tmp_path, "final", expected=cfg)
    assert saved["resume_state"]["epoch"] == 3
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert (tmp_path / "model_3.pth").is_file()
