"""Native-protocol synthetic fixtures for four read-only routing visualizers."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from scipy.io import savemat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_states as taskvis


def make_fixture(root, task, height=9, width=7):
    from linearno_loop.v5.config import resolve_config
    from linearno_loop.v5.schema import make_metadata, write_metadata, EXTERNAL
    from cdlno.linearno_loop.v5.construction import build_from_config
    from cdlno.linearno_loop.v5.checkpoint import _synthetic_metadata, save_pair
    from cdlno.linearno.schema import normalizer_record
    root.mkdir(parents=True, exist_ok=True)
    run, data = root / "saved run with spaces", root / "data with spaces"
    run.mkdir(); data.mkdir()
    config = resolve_config(task, options=dict(architecture="partial_share_feature_gate_v5",
        topology_preset="p1_c3_r2_s1", expert_count={"darcy": 1, "elasticity": 4, "ns": 3, "pipe": 3}[task]),
        profile_overrides={"model.hidden": 8, "model.heads": 2, "model.H": height, "model.W": width})
    model = build_from_config(config).eval()
    for name in taskvis.FILES[task]:
        (data / name).parent.mkdir(parents=True, exist_ok=True)
    total = 1207  # distinguishes native last200 from truncate-first1200/last200
    x, y = np.meshgrid(np.linspace(0, 1, width, dtype=np.float32), np.linspace(0, 1, height, dtype=np.float32))
    if task == "darcy":
        a = np.arange((height-1)*5+1, dtype=np.float32)[:, None] + .1*np.arange((width-1)*5+1, dtype=np.float32)[None, :]
        coeff = np.stack([a+i for i in range(200)])
        target = coeff * .2 + 7
        savemat(data / taskvis.FILES[task][0], dict(coeff=coeff[:1], sol=target[:1]))
        savemat(data / taskvis.FILES[task][1], dict(coeff=coeff, sol=target))
    elif task == "ns":
        u = np.zeros((total, height, width, 20), dtype=np.float32)
        for index in (1000, 1007, 1008):
            u[index] = index * .001 + x[..., None] + .1*y[..., None] + np.arange(20, dtype=np.float32)*.02
        savemat(data / taskvis.FILES[task][0], dict(u=u))
    elif task == "pipe":
        xx = np.zeros((total,height,width),dtype=np.float32); yy = xx.copy()
        u = np.zeros((total,2,height,width),dtype=np.float32)
        for index in (1000,1001,1007):
            xx[index] = x + .02*y**2 + index*.001
            yy[index] = y*.3 + .1*x**2
            u[index,0] = 2*x + y + index*.001
            u[index,1] = -99  # wrong channel must be detected by the expected value
        for name, a in zip(taskvis.FILES[task], (xx,yy,u)):np.save(data/name,a)
    else:
        theta=np.linspace(0,2*np.pi,32,endpoint=False,dtype=np.float32)
        xy=np.zeros((64,2,total),dtype=np.float32); sigma=np.zeros((64,total),dtype=np.float32)
        nodes=np.concatenate([np.stack((.5+r*np.cos(theta),.5+r*np.sin(theta)),-1) for r in (.25,.48)])
        for index in (1000,1007,1008):
            xy[:,:,index]=nodes + (index-1007)*.001
            sigma[:,index]=nodes[:,0]*2+nodes[:,1]+index*.001
        np.save(data/taskvis.FILES[task][0],xy);np.save(data/taskvis.FILES[task][1],sigma)
    checksums={n:taskvis.vis.sha256(data/n) for n in taskvis.FILES[task]}
    raw_states={}
    for key in taskvis.NORMALIZERS[task]:
        if key=='input' and task=='pipe':
            state=dict(mean=torch.tensor([[[.5,.1]]]),std=torch.tensor([[[2.,.25]]]))
        elif key=='input':state=dict(mean=torch.tensor([[3.]]),std=torch.tensor([[2.]]))
        else:state=dict(mean=torch.tensor([[7.]]),std=torch.tensor([[2.5]]))
        raw_states[key]=state
    records={k:normalizer_record(v,fit_split='train only',data_checksum=next(iter(checksums.values())),
        algorithm=taskvis.UNIT_ALGORITHM) for k,v in raw_states.items()}
    base=_synthetic_metadata(config,model)
    sections={k:base[k] for k in EXTERNAL}
    protocol=config['profile_spec']['values']['data']
    sections['data_spec']=dict(protocol=protocol,split=protocol['split'],sampling='synthetic native-layout fixture',
        checksums=checksums,scope='synthetic',runtime=dict(ntrain=1000,ntest=200))
    sections['normalizer_spec']=dict(policy='saved_train_fit' if records else 'none',records=records)
    meta=make_metadata(config,**sections)
    write_metadata(run/'architecture.json',meta);save_pair(run,model,meta)
    return run,data,meta,model


class TaskStatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_native_data_indices_channels_grid_and_saved_normalizers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task in taskvis.TASKS:
                with self.subTest(task=task):
                    _,data,meta,_=make_fixture(Path(tmp)/task,task)
                    taskvis.verify_files(task,data,meta)
                    a=taskvis.load_sample(task,data,meta,0)
                    if task=='darcy':
                        expected=np.arange(9,dtype=np.float32)[:,None]*5 + np.arange(7,dtype=np.float32)[None,:]*.5
                        np.testing.assert_array_equal(a['extras']['coefficient'],expected)
                        np.testing.assert_array_equal(a['fx'].numpy(),((expected-3)/2).reshape(1,-1,1))
                        np.testing.assert_allclose(a['target'],expected*.2+7,rtol=0,atol=1e-6)
                        self.assertEqual(a['raw_sample_index'],0)
                    elif task=='pipe':
                        xx=np.load(data/'Pipe_X.npy')[1000];yy=np.load(data/'Pipe_Y.npy')[1000]
                        np.testing.assert_array_equal(a['x'],xx);np.testing.assert_array_equal(a['y'],yy)
                        expected=(np.stack((xx,yy),-1)-np.array([.5,.1],dtype=np.float32))/np.array([2.,.25],dtype=np.float32)
                        np.testing.assert_array_equal(a['positions'].numpy(),expected.reshape(1,-1,2))
                        np.testing.assert_array_equal(a['target'],np.load(data/'Pipe_Q.npy')[1000,0])
                        self.assertEqual(a['raw_sample_index'],1000)
                    elif task=='elasticity':
                        np.testing.assert_array_equal(a['positions'][0],np.load(data/taskvis.FILES[task][0])[:,:,1007])
                        np.testing.assert_array_equal(a['target'],np.load(data/taskvis.FILES[task][1])[:,1007])
                        self.assertEqual(a['raw_sample_index'],1007)
                    else:
                        from scipy.io import loadmat
                        expected=loadmat(data/taskvis.FILES[task][0])['u'][1007]
                        np.testing.assert_array_equal(a['fx'][0],expected[...,:10].reshape(-1,10))
                        np.testing.assert_array_equal(a['target'],expected[...,10:].reshape(-1,10))
                        self.assertEqual(a['raw_sample_index'],1007)
                    if task!='ns':
                        z=torch.tensor([[0.,1.,2.]])
                        np.testing.assert_array_equal(taskvis.transform(z,a['normalizers']['output'],inverse=True),[[7.,9.5,12.]])

    def test_static_predictions_match_direct_native_forward_and_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task in ('darcy','elasticity','pipe'):
                with self.subTest(task=task):
                    _,data,meta,model=make_fixture(Path(tmp)/task,task)
                    a=taskvis.load_sample(task,data,meta,0)
                    with torch.inference_mode():
                        direct=model(a['positions'],fx=a['fx'])[0,:,0].numpy()
                    q,k,t,p,report,extras=taskvis.infer(task,model,a,torch.device('cpu'),None)
                    np.testing.assert_array_equal(p,direct*2.5+7.)
                    np.testing.assert_array_equal(extras['encoded_prediction'],direct)
                    self.assertEqual(q.shape[-1],64)
                    np.testing.assert_allclose(q.sum(-1),1,atol=5e-6)
                    np.testing.assert_allclose(k.sum(-2),1,atol=5e-6)
                    self.assertTrue(report['hooked_prediction_bitwise_equal'])

    def test_ns_full_prediction_feedback_and_selected_step_match_independent_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,data,meta,model=make_fixture(Path(tmp),'ns')
            a=taskvis.load_sample('ns',data,meta,0)
            current=a['fx'].clone();expected=[];histories=[]
            with torch.inference_mode():
                for _ in range(10):
                    histories.append(current.clone())
                    pred=model(a['positions'],fx=current)
                    expected.append(pred[0,:,0].numpy().copy())
                    current=torch.concat([current[:,:,1:],pred],axis=2)
            expected=np.stack(expected,-1)
            rng=torch.get_rng_state().clone()
            for selected in (1,6,10):
                q,k,t,p,report,extras=taskvis.infer('ns',model,a,torch.device('cpu'),selected)
                np.testing.assert_array_equal(extras['predicted_rollout'],expected)
                np.testing.assert_array_equal(extras['selected_input_history'],histories[selected-1][0])
                np.testing.assert_array_equal(p,expected[:,selected-1])
                np.testing.assert_array_equal(t,a['target'][:,selected-1])
                self.assertEqual(q.shape[-1],32)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            # Changing future ground truth cannot affect weights, inputs or rollout.
            changed=dict(a,target=a['target']+999)
            q2,k2,_,p2,_,e2=taskvis.infer('ns',model,changed,torch.device('cpu'),10)
            np.testing.assert_array_equal(q2,q);np.testing.assert_array_equal(k2,k)
            np.testing.assert_array_equal(p2,p);np.testing.assert_array_equal(e2['predicted_rollout'],expected)
            self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_gpu_fx_capture_and_ns_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for task in ('darcy','ns'):
                _,data,meta,model=make_fixture(Path(tmp)/task,task)
                a=taskvis.load_sample(task,data,meta,0)
                q,k,t,p,report,_=taskvis.infer(task,model.cuda(),a,torch.device('cuda'),10 if task=='ns' else None)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(report['torch_rng_unchanged'])

    def test_conflicts_and_corrupt_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run,data,meta,_=make_fixture(Path(tmp),'pipe')
            args=taskvis.parser('pipe').parse_args(['--run-dir',str(run),'--data-path',str(data)])
            taskvis.validate_request('pipe',args,meta)
            for key,value in [('head','2'),('sample_index',200),('forecast_step',1),('columns',64),('width',float('nan'))]:
                bad=copy.copy(args);setattr(bad,key,value)
                with self.assertRaises(ValueError):taskvis.validate_request('pipe',bad,meta)
            bad=copy.deepcopy(meta);bad['normalizer_spec']['records'].pop('input')
            with self.assertRaisesRegex(ValueError,'normalizer'):taskvis.validate_request('pipe',args,bad)
            with (data/'Pipe_X.npy').open('ab') as f:f.write(b'tamper')
            with self.assertRaisesRegex(ValueError,'checksum'):taskvis.verify_files('pipe',data,meta)

    def test_elasticity_draws_only_nodes_without_invented_triangles(self):
        import matplotlib
        matplotlib.use('Agg',force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        nodes=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        fig,ax=plt.subplots()
        try:
            artist=taskvis.point_paint(ax,nodes,np.arange(4),((0,1),(0,1)),cmap='viridis',norm=Normalize(0,3))
            np.testing.assert_array_equal(artist.get_offsets(),nodes)
            self.assertEqual(len(ax.collections),1)
            self.assertEqual(type(artist).__name__,'PathCollection')
        finally:plt.close(fig)

    def test_four_external_runs_fresh_process_preview_and_export(self):
        with tempfile.TemporaryDirectory(prefix='weight multi-task ') as tmp:
            for task in taskvis.TASKS:
                with self.subTest(task=task):
                    root=Path(tmp)/task;run,data,meta,_=make_fixture(root,task)
                    before={str(p):taskvis.vis.sha256(p) for d in (run,data) for p in d.rglob('*') if p.is_file()}
                    code=("import runpy,sys;sys.path.insert(0,"+repr(str(HERE))+");sys.argv="+
                        repr([str(HERE/'task_states.py'),task,'--run-dir',str(run),'--data-path',str(root/'missing'),'--preview'])+
                        ";runpy.run_path(sys.argv[0],run_name='__main__')")
                    script="import sys\ntry:\n "+code+"\nexcept SystemExit as e:\n assert e.code == 0\nassert 'torch' not in sys.modules\n"
                    subprocess.run([sys.executable,'-B','-c',script],cwd=tmp,check=True,capture_output=True,text=True)
                    output=root/'new figures with spaces'
                    command=[sys.executable,str(HERE/'task_states.py'),task,'--run-dir',str(run),'--data-path',str(data),
                        '--output-dir',str(output),'--device','cpu','--dpi','90','--head','1']
                    env={**os.environ,'CDLNO_PYTHON':sys.executable,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                    result=subprocess.run(command,cwd=tmp,env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                    report=json.loads((output/'metadata.json').read_text())
                    self.assertEqual(report['status'],'completed')
                    self.assertEqual(len(report['figures']),4)
                    self.assertEqual(len(list(output.glob('*.pdf'))),5)
                    self.assertEqual(len(list(output.glob('*.png'))),5)
                    with np.load(output/'routing_weights.npz') as arrays:
                        self.assertEqual(arrays['head_indices'].tolist(),[1])
                        self.assertEqual(arrays['Q'].shape[-1],32 if task=='ns' else 64)
                        self.assertEqual(int(arrays['raw_sample_index']),1007 if task in ('ns','elasticity') else 1000 if task=='pipe' else 0)
                    self.assertIn('\\caption{',(output/'paper_figure.tex').read_text())
                    after={str(p):taskvis.vis.sha256(p) for d in (run,data) for p in d.rglob('*') if p.is_file()}
                    self.assertEqual(before,after)
                    repeat=subprocess.run(command,cwd=tmp,env=env,capture_output=True,text=True)
                    self.assertNotEqual(repeat.returncode,0);self.assertIn('already exists',repeat.stderr)

    def test_ns_parent_listing_never_selects_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp)/'ns runs';parent.mkdir()
            run,data,meta,_=make_fixture(Path(tmp)/'fixture','ns')
            for name in ('experiment a','experiment b'):
                (parent/name).mkdir();(parent/name/'architecture.json').write_bytes((run/'architecture.json').read_bytes())
            before=set(parent.rglob('*'))
            code=("import sys;sys.path.insert(0,"+repr(str(HERE))+");import task_states as t;"
                  "assert t.main(['ns','--run-dir',"+repr(str(parent))+",'--list-runs'])==0;assert 'torch' not in sys.modules")
            result=subprocess.run([sys.executable,'-B','-c',code],check=True,capture_output=True,text=True)
            self.assertEqual(len(json.loads(result.stdout)),2)
            result=subprocess.run([sys.executable,str(HERE/'task_states.py'),'ns','--run-dir',str(parent),'--preview'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('no automatic selection',result.stderr)
            self.assertEqual(before,set(parent.rglob('*')))


if __name__=='__main__':unittest.main()
