"""Synthetic scientific/IO checks; no real dataset or trained weights required."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("airfoil_states", HERE / "airfoil_states.py")
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)


def configuration(height=17, width=7, variant="conv_temp"):
    from linearno_loop.v5.config import resolve_config
    return resolve_config("airfoil", options=dict(architecture="partial_share_feature_gate_v5",
        topology_preset="p1_c3_r2_s1", expert_count=4, actual_M=64),
        profile_overrides={"model.hidden": 8, "model.heads": 2,
                           "model.H": height, "model.W": width,
                           "model.linearno_variant": variant})


def mesh(height, width):
    theta = np.linspace(-np.pi, np.pi, height)[:, None]
    radius = np.linspace(0, 1, width)[None, :]
    x = .5 + (.5 + 1.5 * radius) * np.cos(theta)
    y = (.07 + 1.93 * radius) * np.sin(theta)
    return x.astype(np.float32), y.astype(np.float32)


def make_fixture(root, height=17, width=7):
    from cdlno.linearno_loop.v5.construction import build_from_config
    from cdlno.linearno_loop.v5.checkpoint import _synthetic_metadata, save_pair
    from linearno_loop.v5.schema import make_metadata, write_metadata, EXTERNAL
    cfg = configuration(height, width)
    model = build_from_config(cfg).eval()
    run, data = root / "run with spaces", root / "data with spaces"
    run.mkdir(); data.mkdir()
    x, y = mesh(height, width)
    target = (1. + .1 * x - .05 * y).astype(np.float32)
    for name, shape, values in (
            (vis.DATA_FILES[0], (1200, height, width), x),
            (vis.DATA_FILES[1], (1200, height, width), y),
            (vis.DATA_FILES[2], (1200, 5, height, width), target)):
        a = np.lib.format.open_memmap(data / name, mode="w+", dtype=np.float32, shape=shape)
        if a.ndim == 4:
            a[1000, 4] = values
            a[1001, 4] = values + .1
        else:
            a[1000] = values
            a[1001] = values + .01
        a.flush(); del a
    base = _synthetic_metadata(cfg, model)
    sections = {k: base[k] for k in EXTERNAL}
    sections["data_spec"] = dict(protocol=cfg["profile_spec"]["values"]["data"],
        split="first1000 train; next200 test", sampling="synthetic native-grid fixture",
        checksums={name: vis.sha256(data / name) for name in vis.DATA_FILES},
        scope="synthetic", runtime=dict(ntrain=1000, ntest=200))
    meta = make_metadata(cfg, **sections)
    write_metadata(run / "architecture.json", meta)
    save_pair(run, model, meta)
    return run, data, meta, model, (x, y, target)


class AirfoilStatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_capture_independent_scalar_oracle_and_transparency(self):
        from cdlno.linearno_loop.v5.construction import build_from_config
        for variant in ("plain", "temp", "conv", "conv_temp"):
            with self.subTest(variant=variant):
                model = build_from_config(configuration(3, 4, variant)).eval()
                route = model.loop.suffix[-1].Attn.visits[0]
                if hasattr(route, "temperature_q"):
                    with torch.no_grad():
                        route.temperature_q[:, 0].fill_(.003)  # exercise lower clamp
                        route.temperature_q[:, 1].fill_(.8)
                        route.temperature_k[:, 0].fill_(.09)
                        route.temperature_k[:, 1].fill_(1.7)  # exercise upper clamp
                x = torch.randn(1, 12, 2)
                observed = []
                handle = route.to_q.register_forward_pre_hook(lambda mod, inputs: observed.append(inputs[0].detach().clone()))
                with torch.inference_mode():
                    model(x, fx=None)
                handle.remove()
                q, k, pred, report = vis.capture_last_attention(model, x)
                features = observed[0][0].double().numpy()
                for h in range(2):
                    tq = float(route.temperature_q[0, h, 0, 0].clamp(.01, 1.).detach()) if hasattr(route, "temperature_q") else 1.
                    tk = float(route.temperature_k[0, h, 0, 0].clamp(.01, 1.).detach()) if hasattr(route, "temperature_k") else 1.
                    wq = route.to_q.weight.detach().double().numpy()
                    wk = route.to_k.weight.detach().double().numpy()
                    for point in (0, 5):
                        qs = [sum(float(a * b) for a, b in zip(features[h, point], row)) / tq for row in wq]
                        qe = [math.exp(z - max(qs)) for z in qs]
                        for m in (0, 63):
                            self.assertAlmostEqual(float(q[h, point, m]), qe[m] / sum(qe), delta=2e-6)
                            ks = [sum(float(a * b) for a, b in zip(f, wk[m])) / tk for f in features[h]]
                            ke = [math.exp(z - max(ks)) for z in ks]
                            self.assertAlmostEqual(float(k[h, point, m]), ke[point] / sum(ke), delta=2e-6)
                self.assertTrue(report["hooked_prediction_bitwise_equal"])
                self.assertTrue(report["torch_rng_unchanged"])
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_capture(self):
        from cdlno.linearno_loop.v5.construction import build_from_config
        model = build_from_config(configuration(3, 4)).cuda().eval()
        q, k, pred, report = vis.capture_last_attention(model, torch.randn(1, 12, 2, device="cuda"))
        self.assertTrue(np.isfinite(q).all() and np.isfinite(k).all() and np.isfinite(pred).all())
        self.assertTrue(report["torch_rng_unchanged"])

    def test_peak_view_does_not_turn_uniform_noise_into_patterns(self):
        weights = np.full((10, 64), .015625)
        peak, vmax, _ = vis.display_values(weights, "Q", "peak")
        np.testing.assert_array_equal(peak, np.ones_like(weights))
        values, _, _ = vis.display_values(weights, "K", "shared")
        np.testing.assert_array_equal(values, weights * 10)
        zero, _, _ = vis.display_values(np.zeros((3, 64)), "Q", "peak")
        np.testing.assert_array_equal(zero, 0)

    def test_mesh_connectivity_does_not_fill_airfoil_hole(self):
        x, y = mesh(65, 9)
        triangles = vis.structured_mesh(x, y)
        nodes = triangles.triangles
        # Every triangle is contained in exactly one adjacent structured cell.
        self.assertTrue(np.all(np.ptp(nodes // 9, axis=1) == 1))
        self.assertTrue(np.all(np.ptp(nodes % 9, axis=1) == 1))
        cent_x = x.ravel()[nodes].mean(1)
        cent_y = y.ravel()[nodes].mean(1)
        self.assertTrue(np.all(((cent_x - .5) / .5)**2 + (cent_y / .07)**2 > .995))

    def test_data_sample_channel_checksums_and_request_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, data, meta, model, (x, y, target) = make_fixture(Path(tmp))
            xx, yy, tt, _ = vis.load_sample(data, meta, 0)
            np.testing.assert_array_equal(xx, x); np.testing.assert_array_equal(yy, y)
            np.testing.assert_array_equal(tt, target)
            _, _, next_target, _ = vis.load_sample(data, meta, 1)
            np.testing.assert_array_equal(next_target, target + .1)
            args = vis.parser().parse_args(["--run-dir", str(run), "--data-path", str(data)])
            for changes in ({"sample_index": 200}, {"head": "2"}, {"near_xlim": [1., 0.]}):
                altered = copy.copy(args)
                for k, v in changes.items():setattr(altered, k, v)
                with self.assertRaises(ValueError):vis.validate_request(altered, meta)
            changed = copy.deepcopy(meta); changed["resolved_config"]["actual_M"] = 32
            with self.assertRaisesRegex(ValueError, "64-state"):
                vis.validate_request(args, changed)
            with (data / vis.DATA_FILES[0]).open("ab") as f:f.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                vis.load_sample(data, meta, 0)

    def test_new_process_strict_checkpoint_and_full_plot_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, data, meta, model, _ = make_fixture(root)
            before = {p.relative_to(run): vis.sha256(p) for p in run.rglob("*") if p.is_file()}
            output = root / "figures with spaces"
            # Preview explicitly verifies no Torch import and no data requirement.
            code = ("import runpy,sys; sys.argv=" + repr([str(HERE / "airfoil_states.py"),
                "--run-dir", str(run), "--data-path", str(root / "missing"), "--preview"]) +
                "; runpy.run_path(sys.argv[0],run_name='__main__')")
            # main raises SystemExit; inspect imports after catching its successful exit.
            preview = "import sys\ntry:\n " + code + "\nexcept SystemExit as e:\n assert e.code == 0\nassert 'torch' not in sys.modules\n"
            subprocess.run([sys.executable, "-B", "-c", preview], cwd=tmp, check=True, capture_output=True, text=True)
            command = [sys.executable, str(HERE / "airfoil_states.py"), "--run-dir", str(run), "--data-path", str(data),
                       "--device", "cpu", "--output-dir", str(output), "--dpi", "90"]
            result = subprocess.run(command, cwd=tmp, check=True, capture_output=True, text=True,
                                    env={**__import__('os').environ, "CDLNO_PYTHON": sys.executable,
                                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            report = json.loads((output / "metadata.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["figures"]), 8)
            self.assertEqual(len(list(output.glob("*.pdf"))), 9)
            self.assertEqual(len(list(output.glob("*.png"))), 9)
            with np.load(output / "routing_weights.npz") as a:
                self.assertEqual(a["Q"].shape, (1, 17 * 7, 64))
                np.testing.assert_allclose(a["Q"].sum(-1), 1., atol=5e-6)
                np.testing.assert_allclose(a["K"].sum(-2), 1., atol=5e-6)
                self.assertEqual(a["raw_sample_index"], 1000)
            after = {p.relative_to(run): vis.sha256(p) for p in run.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            failed = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("already exists", failed.stderr)
            # Native strict pair validation rejects tampered weights.
            weights = run / "weights/epoch_0001.pt"
            with weights.open("ab") as stream:stream.write(b"bad")
            from cdlno.linearno_loop.v5.checkpoint import load_model
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(run)


if __name__ == "__main__":
    unittest.main()
