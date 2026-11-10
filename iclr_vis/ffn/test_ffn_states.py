"""Independent expert-math and real synthetic checkpoint/CLI acceptance."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from scipy.special import erf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ffn_states as ffn
from capture import capture_experts, summarize
sys.path.insert(0, str(HERE.parent/"weight"))
import test_airfoil_states as air_fixture
import test_task_states as task_fixture


def nonuniform(model):
    with torch.no_grad():
        for group in (model.loop.prefix, model.loop.core, model.loop.suffix):
            for block in group:
                for visit, route in enumerate(block.visits):
                    w = torch.arange(route.router.weight.numel(), device=route.router.weight.device).reshape_as(route.router.weight)
                    route.router.weight.copy_(torch.sin(w + visit*2) * .4)
                    route.router.bias.copy_(torch.linspace(-.5, .5, len(route.router.bias), device=route.router.bias.device) * (visit+1))


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_numpy_expert_math_and_visit_ownership(self):
        from cdlno.linearno_loop.v5.construction import build_from_config
        for variant in ("plain", "temp", "conv", "conv_temp"):
            with self.subTest(variant=variant):
                model = build_from_config(air_fixture.configuration(3,4,variant)).eval()
                nonuniform(model)
                x = torch.randn(1,12,2)
                samples, handles = [], []
                for group in (model.loop.prefix, model.loop.core, model.loop.suffix):
                    for block in group:
                        for visit, route in enumerate(block.visits):
                            def record(module, args, block=block, route=route, visit=visit):
                                samples.append((block, route, visit, args[0].detach().cpu().numpy().copy()))
                            handles.append(route.router.register_forward_pre_hook(record))
                with torch.inference_mode():
                    expected_prediction = model(x, fx=None)
                for handle in handles: handle.remove()
                initial = {n:t.clone() for n,t in model.state_dict().items()}
                arrays, rows, prediction, report = capture_experts(model,x)
                np.testing.assert_array_equal(prediction,expected_prediction[0,:,0])
                self.assertEqual([(r['group'],r['position'],r['visit']) for r in rows],
                    [('prefix',0,0),('core',0,0),('core',1,0),('core',2,0),('core',0,1),('core',1,1),('core',2,1),('suffix',0,0)])
                for j,(block,route,visit,u) in enumerate(samples):
                    # Independent NumPy linear algebra and analytic GELU; no production forward/helper.
                    logits = u.astype(np.float64) @ route.router.weight.detach().numpy().T + route.router.bias.detach().numpy()
                    mass = np.exp(logits-logits.max(-1,keepdims=True))
                    pi = mass/mass.sum(-1,keepdims=True)
                    np.testing.assert_allclose(arrays['probabilities'][j],pi[0],atol=1e-7,rtol=1e-6)
                    for e, expert in enumerate(block.experts):
                        first,last = expert.linear_pre[0],expert.linear_post
                        hidden = u.astype(np.float64) @ first.weight.detach().numpy().T + first.bias.detach().numpy()
                        hidden = .5*hidden*(1+erf(hidden/np.sqrt(2)))
                        value = hidden @ last.weight.detach().numpy().T + last.bias.detach().numpy()
                        expected_norm = np.linalg.norm(rows[j]['expert_scale'] * pi[...,e:e+1] * value, axis=-1)[0]
                        np.testing.assert_allclose(arrays['contribution_norm'][j,:,e],expected_norm,atol=1e-7,rtol=2e-5)
                self.assertFalse(np.array_equal(arrays['probabilities'][1],arrays['probabilities'][4]))
                self.assertTrue(report['hooked_prediction_bitwise_equal'])
                for n,t in model.state_dict().items():self.assertTrue(torch.equal(t,initial[n]))
                self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_known_probabilities_entropy_and_no_peak_rescaling(self):
        arrays = dict(probabilities=np.array([[[.25,.75],[.5,.5]]]), contribution_norm=np.ones((1,2,2)),
                      expert_output_norm=np.ones((1,2,2)),mixed_update_norm=np.ones((1,2)))
        summary = summarize(arrays,[dict(group='core',position=0,visit=0,logical_index=0,expert_scale=.5)])
        self.assertEqual(summary['visits'][0]['mean_gate'],[.375,.625])
        np.testing.assert_allclose(arrays['normalized_entropy'],[[.8112781244591328,1]],atol=1e-14)
        np.testing.assert_array_equal(arrays['probabilities'],[[[.25,.75],[.5,.5]]])

    def test_custom_topology_eight_experts_three_visits_shared_norms(self):
        from linearno_loop.v5.config import resolve_config
        from cdlno.linearno_loop.v5.construction import build_from_config
        cfg=resolve_config('airfoil',options=dict(architecture='partial_share_feature_gate_v5',
            topology_preset='custom',prefix_blocks=2,recurrent_core_blocks=2,loop_repeats=3,suffix_blocks=2,
            expert_count=8,core_norm_mode='shared'),profile_overrides={
                'model.hidden':8,'model.heads':2,'model.H':3,'model.W':4})
        model=build_from_config(cfg).eval();nonuniform(model)
        arrays,rows,_,report=capture_experts(model,torch.randn(1,12,2))
        self.assertEqual(arrays['probabilities'].shape,(10,12,8))
        self.assertEqual(report['expert_calls'],80)
        self.assertEqual([r['expert_scale'] for r in rows],[1.,1.]+[1/3]*6+[1.,1.])

    def test_single_expert_and_uniform_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,data,meta,model = task_fixture.make_fixture(Path(tmp),'darcy')
            sample = task_fixture.taskvis.load_sample('darcy',data,meta,0)
            arrays, rows, _, _ = capture_experts(model,sample['positions'],fx=sample['fx'])
            np.testing.assert_array_equal(arrays['probabilities'],np.ones((8,63,1)))
            summary = summarize(arrays,rows)
            self.assertNotIn('normalized_entropy',arrays)
            self.assertIsNone(summary['visits'][0]['mean_normalized_entropy'])
            self.assertGreater(arrays['contribution_norm'].max(),0)
        from cdlno.linearno_loop.v5.construction import build_from_config
        model = build_from_config(air_fixture.configuration(3,4)).eval()
        arrays,_,_,_ = capture_experts(model,torch.randn(1,12,2))
        np.testing.assert_array_equal(arrays['probabilities'],np.full((8,12,4),.25))

    def test_observer_catches_corrupted_residual_and_removes_hooks(self):
        from cdlno.linearno_loop.v5.construction import build_from_config
        model = build_from_config(air_fixture.configuration(3,4)).eval()
        # Deliberately corrupt a real block output to prove the residual check detects it.
        bad = model.loop.core[0].register_forward_hook(lambda m,i,o:o+.1)
        try:
            with self.assertRaises(AssertionError):capture_experts(model,torch.randn(1,12,2))
        finally:bad.remove()
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        with torch.no_grad():model.loop.core[0].visits[0].router.bias[0] = float('nan')
        with self.assertRaisesRegex(ValueError,'NaN/Inf'):capture_experts(model,torch.randn(1,12,2))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_ns_prediction_feedback_and_native_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task in ('darcy','elasticity','ns','pipe'):
                with self.subTest(task=task):
                    _,data,meta,model = task_fixture.make_fixture(Path(tmp)/task,task)
                    nonuniform(model)
                    sample = task_fixture.taskvis.load_sample(task,data,meta,0)
                    arrays,rows,target,pred,report,extras = ffn.infer(task,model,sample,torch.device('cpu'),6 if task=='ns' else None)
                    with torch.inference_mode():
                        if task == 'ns':
                            current=sample['fx'].clone(); expected=[]
                            for step in range(10):
                                if step==5:np.testing.assert_array_equal(extras['selected_input_history'],current[0])
                                out=model(sample['positions'],fx=current)
                                expected.append(out[0,:,0].numpy())
                                current=torch.cat((current[...,1:],out),-1)
                            np.testing.assert_array_equal(extras['predicted_rollout'],np.stack(expected,-1))
                            np.testing.assert_array_equal(pred,expected[5])
                            np.testing.assert_array_equal(target,sample['target'][:,5])
                        else:
                            out=model(sample['positions'],fx=sample['fx'])[0,:,0].numpy()
                            np.testing.assert_array_equal(pred,out*2.5+7.)
                    self.assertTrue(report['hooked_prediction_bitwise_equal'])

    def test_preview_no_tensor_import_and_error_propagation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            run,data,_,_,_=air_fixture.make_fixture(root)
            code = "import sys;sys.path.insert(0,sys.argv.pop(1));import ffn_states as f;f.main(sys.argv[1:]);assert 'torch' not in sys.modules and 'matplotlib' not in sys.modules"
            result=subprocess.run([sys.executable,'-B','-c',code,str(HERE),'airfoil','--run-dir',str(run),
                '--data-path',str(root/'does not exist'),'--preview'],capture_output=True,text=True,cwd=tmp)
            self.assertEqual(result.returncode,0,result.stderr)
            for flags in (['--sample-index','200'],['--forecast-step','1'],['--font-size','nan']):
                result=subprocess.run([sys.executable,str(HERE/'ffn_states.py'),'airfoil','--run-dir',str(run),'--preview',*flags],
                    capture_output=True,text=True,env={**os.environ,'CDLNO_PYTHON':sys.executable})
                self.assertNotEqual(result.returncode,0)
            result=subprocess.run([sys.executable,str(HERE/'ffn_states.py'),'ns','--run-dir',str(root),'--list-runs'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),[])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_nonuniform_gate_capture(self):
        from cdlno.linearno_loop.v5.construction import build_from_config
        model=build_from_config(air_fixture.configuration(3,4)).cuda().eval();nonuniform(model)
        arrays,rows,_,report=capture_experts(model,torch.randn(1,12,2,device='cuda'))
        self.assertEqual(arrays['probabilities'].shape,(8,12,4))
        self.assertTrue(report['public_rng_unchanged'])
        self.assertEqual(report['expert_calls'],32)


if __name__ == '__main__': unittest.main()
