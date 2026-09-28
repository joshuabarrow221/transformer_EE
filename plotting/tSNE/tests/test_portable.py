"""Exercise real forward passes, selection, relocation, and cache rejection.

The tiny randomly initialized network verifies mechanics, not physics quality.
Set TRANSFORMEREE_REPO when this package is being tested outside its checkout.
"""
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from run import select_source, resolve, sha256
from topology import decode
from inference import load_network, forward_fast, forward_frame

REPO = Path(os.environ.get('TRANSFORMEREE_REPO', HERE.parents[1]))
sys.path.insert(0,str(REPO))
from transformer_ee.model.transformerEncoder import Transformer_EE_MV


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        model = self.root/'models/tiny'
        model.mkdir(parents=True)
        (self.root/'samples').mkdir()
        self.config = dict(vector=['Final_State_Particles_PDG','Final_State_Particles_Energy'],
            scalar=['tot_fKE'], target=['Nu_Energy','Nu_Mom_X','Nu_Mom_Y','Nu_Mom_Z','Topology'],
            max_num_prongs=4, model=dict(name='Transformer_EE_MV',kwargs=dict(
                d_model=8,nhead=2,num_layers=1,dim_feedforward=16,dropout=0.0)),
            loss=dict(kwargs=dict(coefficients=[1,1,1,1,0])))
        self.stats = {v:[0.0,1.0] for v in self.config['vector']+self.config['scalar']}
        self.stats['Final_State_Particles_PDG'] = [0.0,1000.0]
        (model/'input.json').write_text(json.dumps(self.config))
        (model/'trainset_stat.json').write_text(json.dumps(self.stats))
        torch.manual_seed(8)
        torch.save(Transformer_EE_MV(self.config).state_dict(), model/'best_model.zip')
        rows = []
        for i in range(65):
            p, pi = i%4, i%2
            pdgs = [13]+[2212]*p+[111]*pi
            rows.append(dict(Event_Index=100+i,Topology=f'3{p:02d}0000000000{pi:02d}',
                Final_State_Particles_PDG=','.join(map(str,pdgs)),
                Final_State_Particles_Energy=','.join(str(.2+.01*i) for _ in pdgs),
                tot_fKE=.5+.01*i,Nu_Energy=1+.01*i,Nu_Mom_X=0,Nu_Mom_Y=0,Nu_Mom_Z=1))
        self.frame = pd.DataFrame(rows)
        # One deliberately invalid input is excluded independently of topology.
        self.frame.loc[3,'Final_State_Particles_Energy'] = 'nan'
        self.frame.to_csv(self.root/'samples/events.csv',index=False)
        self.manifest = dict(version=1,models={'tiny':dict(directory='tiny',sha256={
            f:sha256(model/f) for f in ['best_model.zip','input.json','trainset_stat.json']})},
            sources={'beam':dict(path='events.csv',generator='AR23')},
            groups=[dict(id='tiny',title='Tiny integration test',training_generator='AR23',
                         rows={'Flat_NoNoise':dict(models=['tiny'],sources=['beam'])})])
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def command(self, *extra, success=True):
        command = [sys.executable,str(HERE/'run.py'),'--manifest',str(self.root/'manifest.json'),
                   '--repo',str(REPO),'--models-root',str(self.root/'models'),
                   '--samples-root',str(self.root/'samples'),'--output',str(self.root/'result'),
                   '--events','48','--batch-size','8','--chunk-size','17','--threads','1',
                   '--perplexity','5','--iterations','300',*extra]
        result = subprocess.run(command,capture_output=True,text=True)
        if success:
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0)
        return result

    def test_selection_and_reference_forward(self):
        frame,audit = select_source({'path':str(self.root/'samples/events.csv')},
                                   {'tiny':{'config':self.config}},48)
        self.assertEqual(len(frame),48)
        self.assertNotIn(3,frame.source_row.to_numpy())
        self.assertEqual(audit['excluded_before_last_selected'],1)
        net,config,stats = load_network(REPO,self.root/'models/tiny')
        torch.set_num_threads(1)
        fast = forward_fast(net,config,stats,frame,batch_size=8)
        reference = forward_frame(net,config,stats,frame,batch_size=8)
        for a,b in zip(fast[:3],reference[:3]):
            np.testing.assert_allclose(a,b,rtol=2e-4,atol=1e-5)
        self.assertGreater(fast[3]['truncated_events'],0)

    @unittest.skipUnless(importlib.util.find_spec('polars'),'Optional Polars is not installed')
    def test_polars_normalization_padding_and_full_export(self):
        frame,_ = select_source({'path':str(self.root/'samples/events.csv')},
                                {'tiny':{'config':self.config}},48)
        net,config,stats = load_network(REPO,self.root/'models/tiny')
        torch.set_num_threads(1)
        reference = forward_fast(net,config,stats,frame,batch_size=8)
        alternate = forward_fast(net,config,stats,frame,batch_size=8,prepare_engine='polars')
        for a,b in zip(reference[:3],alternate[:3]):
            np.testing.assert_allclose(a,b,rtol=2e-4,atol=1e-5)
        self.assertEqual(reference[3]['truncated_events'],alternate[3]['truncated_events'])
        self.command('--stage','extract','--prepare-engine','polars')
        meta = json.loads((self.root/'result/features/tiny_Flat_NoNoise_beam.json').read_text())
        self.assertEqual(meta['prepare_engine'],'polars')

    def test_complete_resume_and_relocated_plot(self):
        self.command('--stage','all')
        feature = self.root/'result/features/tiny_Flat_NoNoise_beam.npz'
        before = feature.stat().st_mtime_ns
        self.command('--stage','extract')
        self.assertEqual(before,feature.stat().st_mtime_ns)
        with np.load(feature,allow_pickle=False) as data:
            self.assertEqual(data['features'].shape[0],48)
            self.assertEqual(data['prediction'].shape,(48,4))
            self.assertEqual(len(set(data['event_index'])),48)
        # Rendering is deliberately tested with moved outputs and unavailable
        # original assets, as on a CPU host receiving GPU-extracted features.
        (self.root/'result').rename(self.root/'moved')
        (self.root/'models').rename(self.root/'models-unavailable')
        (self.root/'samples').rename(self.root/'samples-unavailable')
        result = subprocess.run([sys.executable,str(HERE/'run.py'),'--stage','plot',
            '--output',str(self.root/'moved'),'--threads','1','--perplexity','5',
            '--iterations','300'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertTrue((self.root/'moved/plots/tiny/grid.pdf').exists())

    def test_checkpoint_mutation_rejected(self):
        path = self.root/'models/tiny/input.json'
        path.write_text(path.read_text()+'\n')
        result = self.command('--stage','check',success=False)
        self.assertIn('hash mismatch',result.stderr)

    def test_duplicate_target_rejected(self):
        self.manifest['groups'][0]['rows']['Flat_NoNoise']['models'] = ['tiny','tiny']
        (self.root/'manifest.json').write_text(json.dumps(self.manifest))
        result = self.command('--stage','check',success=False)
        self.assertIn('each physical target once',result.stderr)

    def test_asset_copy_and_existing_file_protection(self):
        assets = []
        for name in ['best_model.zip','input.json','trainset_stat.json']:
            path = self.root/'models/tiny'/name
            assets.append(dict(kind='model',source='models/tiny/'+name,
                               destination='tiny/'+name,bytes=path.stat().st_size,sha256=sha256(path)))
        sample = self.root/'samples/events.csv'
        assets.append(dict(kind='sample',source='samples/events.csv',destination='events.csv',bytes=sample.stat().st_size))
        inventory = self.root/'private-inventory.json'
        inventory.write_text(json.dumps(dict(version=1,assets=assets)))
        command = [sys.executable,str(HERE/'stage_assets.py'),'--manifest',str(self.root/'manifest.json'),
            '--inventory',str(inventory),'--source-root',str(self.root),
            '--destination',str(self.root/'transfer'),'--copy']
        result = subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        copied = self.root/'transfer/samples/events.csv'
        self.assertEqual(sha256(copied),sha256(sample))
        copied.write_text('different contents')
        result = subprocess.run(command,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Refusing to overwrite',result.stderr)

    @unittest.skipIf(torch.cuda.is_available(),'Requires a CPU-only host')
    def test_cuda_request_never_silently_falls_back(self):
        result = self.command('--stage','check','--device','cuda',success=False)
        self.assertIn('CUDA requested but unavailable',result.stderr)


if __name__ == '__main__':
    unittest.main()
