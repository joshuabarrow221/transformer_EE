"""CPU contract checks plus an opt-in real CUDA integration test.

Mocks verify routing/configuration only; they are never evidence of CUDA
performance or numerical correctness. Set TRANSFORMEREE_TEST_GPU=1 on the A100
for the real cuML smoke test, which fails (rather than skips) on a broken GPU.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from gpu_tsne import parameters
from plot_tsne import fit_embedding
from render import file_hash, load_coordinates, render_all


class GPUContractTests(unittest.TestCase):
    def test_explicit_parameters_and_limits(self):
        values = parameters(999950, 42, 30, 1000)
        self.assertEqual(values['n_neighbors'], 90)
        self.assertEqual(values['perplexity'], 30)
        self.assertEqual(values['learning_rate_method'], 'none')
        self.assertEqual(values['max_iter'], 1000)
        self.assertEqual(values['exaggeration_iter'], 250)
        self.assertEqual(values['init'], 'pca')
        with self.assertRaises(ValueError):
            parameters(999950, 42, 400, 1000)
        with self.assertRaises(ValueError):
            parameters(40, 42, 40, 1000)
        with self.assertRaises(ValueError):
            parameters(100, 42, 5, 300, float('nan'))

    def test_routes_float32_pca_and_device_without_gpu(self):
        matrix = np.random.default_rng(3).normal(size=(80, 64)).astype('float32')
        coords = np.random.default_rng(4).normal(size=(80, 2)).astype('float32')
        with patch('gpu_tsne.runtime'), patch('gpu_tsne.fit',return_value=(coords,1.0,{'backend':'mock cuML'})) as mock:
            result, info = fit_embedding(matrix, 42, 5, 300, backend='cuml', threads=1,
                                         tsne_device=2, gpu_learning_rate=300)
        self.assertEqual(mock.call_args.args[0].shape, (80,50))
        self.assertEqual(mock.call_args.args[0].dtype,np.float32)
        self.assertEqual(mock.call_args.kwargs,dict(device=2,learning_rate=300))
        np.testing.assert_array_equal(result,coords)
        self.assertEqual(info['sample_n'],80)

    def test_adapter_calls_cuml_and_synchronizes_selected_device(self):
        from contextlib import nullcontext
        from types import SimpleNamespace, ModuleType
        from unittest.mock import Mock
        import sys
        from gpu_tsne import fit
        matrix = np.ones((48,4),dtype=np.float32)
        estimator = Mock(kl_divergence_=1.25,n_iter_=300)
        estimator.fit_transform.return_value=np.ones((48,2),dtype=np.float32)
        module = ModuleType('cuml.manifold')
        module.TSNE=Mock(return_value=estimator)
        stream=Mock()
        cp=SimpleNamespace(float32=np.float32,asarray=Mock(side_effect=np.asarray),
            cuda=SimpleNamespace(Device=Mock(return_value=nullcontext()),get_current_stream=lambda:stream))
        with patch('gpu_tsne.runtime',return_value=(cp,SimpleNamespace(__version__='test'),{'device_index':2})), \
             patch.dict(sys.modules,{'cuml.manifold':module}):
            coordinates, divergence, info=fit(matrix,42,5,300,device=2)
        cp.cuda.Device.assert_called_once_with(2)
        stream.synchronize.assert_called_once()
        self.assertEqual(module.TSNE.call_args.kwargs['method'],'fft')
        self.assertEqual(module.TSNE.call_args.kwargs['learning_rate_method'],'none')
        self.assertEqual(coordinates.shape,(48,2))
        self.assertEqual(info['actual_iterations'],300)
        self.assertEqual(divergence,1.25)

    def test_missing_cuml_has_actionable_error(self):
        import sys
        from gpu_tsne import runtime
        with patch.dict(sys.modules,{'cuml':None}):
            with self.assertRaisesRegex(RuntimeError,'No CPU fallback'):
                runtime()

    def test_gpu_failure_does_not_fall_back(self):
        with patch('gpu_tsne.runtime',side_effect=RuntimeError('GPU unavailable')), patch('plot_tsne.PCA') as pca:
            with self.assertRaisesRegex(RuntimeError,'GPU unavailable'):
                fit_embedding(np.ones((80,64)),42,5,300,backend='cuml')
            pca.assert_not_called()

    def test_gpu_joint_fit_resume_and_parameter_invalidation(self):
        from types import SimpleNamespace
        from plot_tsne import run_group
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = dict(id='joint',title='Joint test',entries=[])
            for i, gen in enumerate(['AR23','NuWro']):
                path = root/(gen+'.npz')
                np.savez(path,features=np.random.default_rng(i).normal(size=(48,64)).astype('float32'),
                    source_row=np.arange(48),event_index=np.arange(48)+100*i,
                    topology=np.array(['301000000000000']*48))
                path.with_suffix('.json').write_text(json.dumps(dict(block_sizes=[64])))
                group['entries'].append(dict(row='Flat_NoNoise',generator=gen,latent_path=str(path)))
            args = SimpleNamespace(output=root/'plots',representation='latent',events=48,seed=42,
                perplexity=5,iterations=300,backend='cuml',sampling='first',lazy_latent_load=True,
                strict=True,tsne_device=0,gpu_learning_rate=200.0)
            coords = np.random.default_rng(7).normal(size=(96,2)).astype('float32')
            with patch('gpu_tsne.runtime',return_value=(None,None,{'cuml':'test'})), \
                 patch('gpu_tsne.fit',return_value=(coords,1.0,{'backend':'mock cuML'})) as fit, \
                 patch('plot_tsne.draw_grid'):
                run_group(group,args)
                self.assertEqual(fit.call_args.args[0].shape,(96,50))
                run_group(group,args)
                self.assertEqual(fit.call_count,1)
                args.gpu_learning_rate=300.0
                run_group(group,args)
                self.assertEqual(fit.call_count,2)
                # A finite-but-altered coordinate must not be accepted as a cache hit.
                path=root/'plots/joint/Flat_NoNoise_AR23.csv.gz'
                frame=pd.read_csv(path,dtype={'true_Topology':str,'topology_code':str})
                frame.loc[0,'TSNE1']+=1
                frame.to_csv(path,index=False)
                run_group(group,args)
                self.assertEqual(fit.call_count,3)
            metadata, frames=load_coordinates(root/'plots/joint')
            self.assertEqual(sum(map(len,frames.values())),96)
            self.assertEqual(metadata['fits']['Flat_NoNoise']['sample_n'],96)

    @unittest.skipUnless(os.environ.get('TRANSFORMEREE_TEST_GPU') == '1', 'Opt in on a CUDA/cuML host')
    def test_real_gpu_fit(self):
        from sklearn.datasets import make_blobs
        matrix, _ = make_blobs(n_samples=1500,n_features=64,centers=8,random_state=7)
        coordinates, info = fit_embedding(matrix,42,30,1000,backend='cuml',threads=4)
        self.assertEqual(coordinates.shape,(1500,2))
        self.assertTrue(np.isfinite(coordinates).all())
        self.assertEqual(info['backend'],'cuML FFT')
        self.assertGreater(info['trustworthiness_10nn'],0.85)
        print(json.dumps(info,indent=2))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.grid = self.root/'plots/tiny'
        self.grid.mkdir(parents=True)
        self.csv = self.grid/'Flat_NoNoise_AR23.csv.gz'
        frame = pd.DataFrame(dict(source_row=np.arange(48),true_Topology=['301000000000000']*48,
            category=['1p0pi']*48,TSNE1=np.arange(48),TSNE2=np.arange(48)*.5,
            generator=['AR23']*48,training_condition=['Flat_NoNoise']*48))
        frame.to_csv(self.csv,index=False)
        self.metadata = dict(group=dict(id='tiny',title='Test grid',entries=[dict(row='Flat_NoNoise',generator='AR23')]),
            representation='latent', arguments=dict(seed=42,perplexity=5),
            fits={'Flat_NoNoise':dict(sample_n=48)},
            coordinate_files={self.csv.name:dict(sha256=file_hash(self.csv),rows=48,row='Flat_NoNoise',generator='AR23')})
        self.save()

    def save(self):
        (self.grid/'metadata.json').write_text(json.dumps(self.metadata))

    def tearDown(self):
        self.tmp.cleanup()

    def test_redraw_without_activations_or_gpu(self):
        before = file_hash(self.csv)
        style = self.root/'style.json'
        style.write_text(json.dumps(dict(colors={'1p0pi':'red'},dpi=30,title='New title')))
        with patch('plot_tsne.fit_embedding', side_effect=AssertionError('Must not fit')), patch('numpy.load',side_effect=AssertionError('Must not read NPZ')):
            render_all(self.root/'plots',self.root/'redraw',style_path=style)
        self.assertTrue((self.root/'redraw/tiny/grid.png').exists())
        self.assertTrue((self.root/'redraw/tiny/grid.pdf').exists())
        self.assertEqual(file_hash(self.csv),before)

    def test_reject_changed_coordinates_and_missing_panels(self):
        self.csv.write_bytes(self.csv.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            load_coordinates(self.grid)
        self.csv.unlink()
        with self.assertRaises(FileNotFoundError):
            load_coordinates(self.grid)

    def test_reject_mixed_fit_counts(self):
        self.metadata['fits']['Flat_NoNoise']['sample_n']=49
        self.save()
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            load_coordinates(self.grid)

    def test_historical_export_and_overwrite_protection(self):
        del self.metadata['coordinate_files']
        self.save()
        with self.assertWarns(UserWarning):
            _,frames=load_coordinates(self.grid)
        self.assertEqual(len(frames[('Flat_NoNoise','AR23')]),48)
        with self.assertRaisesRegex(ValueError,'separate'):
            render_all(self.root/'plots',self.root/'plots')


if __name__ == '__main__':
    unittest.main()
