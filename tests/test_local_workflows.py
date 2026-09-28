"""Regression checks for local workflow changes merged with upstream configuration defaults."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'transformer_ee/inference/configuration_file_tools/generate_batch_inference_configs.py'


class LocalWorkflowTests(unittest.TestCase):
    def test_default_generator_from_another_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(GENERATOR), '--outdir', tmp,
                 '--model-search-roots', str(Path(tmp) / 'absent')],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = sorted(Path(tmp).glob('*.json'))
            self.assertEqual(len(files), 3)
            tunes = set()
            for file in files:
                config = json.loads(file.read_text())
                self.assertEqual(len(config['models']), 8)
                self.assertEqual(len(config['pairs']), 8)
                self.assertEqual(len(config['samples']), 2)
                samples = {s['name']: s['path'] for s in config['samples']}
                for pair in config['pairs']:
                    self.assertIn('VectorLeptwNC', samples[pair['sample']])
                tune = file.stem.rsplit('_', 1)[1]
                tunes.add(tune)
                self.assertTrue(all(s['name'].endswith(tune) for s in config['samples']))
            self.assertEqual(tunes, {'G1810a0211a', 'G1810a0211b', 'G2111a'})

    def test_generator_rejects_empty_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / 'empty.txt'
            empty.write_text('')
            result = subprocess.run([sys.executable, str(GENERATOR), '--beam-files', str(empty),
                                     '--outdir', tmp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(list(Path(tmp).glob('*.json')))

    def test_training_config_precedence_and_unique_runs(self):
        captures = []
        logger_calls = []
        train = types.ModuleType('transformer_ee.train')
        class Trainer:
            def __init__(self, config, logger):
                captures.append(config)
            def train_LCL(self):
                pass
            def eval(self):
                pass
        train.MVtrainer = Trainer
        logger = types.ModuleType('transformer_ee.logger.wandb_train_logger')
        logger.WandBLogger = lambda **kwargs: logger_calls.append(kwargs)
        torch = types.ModuleType('torch')
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, ipc_collect=lambda: None)
        modules = {'transformer_ee.train': train,
                   'transformer_ee.logger.wandb_train_logger': logger,
                   'torch': torch, 'wandb': types.ModuleType('wandb')}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('train_wide_check', ROOT / 'train_wide.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config_path = Path(tmp) / 'input.json'
            configured_path = Path(tmp) / 'configured'
            config = {'save_path': str(configured_path), 'model': {'kwargs': {'nhead': 4}},
                      'optimizer': {'name': 'adam'}}
            config_path.write_text(json.dumps(config))
            for _ in range(2):
                argv = ['train_wide.py', '--base-config', str(config_path),
                        '--save-path', str(Path(tmp) / 'cli'), '--auto-unique-save-path',
                        '--num-layers', '7', '--wandb-name', 'x' * 300]
                with patch.object(sys, 'argv', argv):
                    module.main()
            self.assertNotEqual(captures[0]['save_path'], captures[1]['save_path'])
            for cfg, call in zip(captures, logger_calls):
                self.assertEqual(Path(cfg['save_path']).parent, configured_path)
                self.assertEqual(cfg['model']['kwargs']['nhead'], 4)
                self.assertEqual(cfg['model']['kwargs']['num_layers'], 7)
                self.assertEqual(cfg['optimizer']['name'], 'adam')
                self.assertNotIn('id', call)
                self.assertEqual(len(call['name']), 120)
                self.assertTrue((Path(cfg['save_path']) / 'train.log').is_file())
            config['model']['kwargs']['num_layers'] = 9
            config_path.write_text(json.dumps(config))
            with patch.object(sys, 'argv', ['train_wide.py', '--base-config', str(config_path),
                                          '--num-layers', '7', '--wandb-id', 'explicit-id']):
                module.main()
            self.assertEqual(captures[-1]['save_path'], str(configured_path))
            self.assertEqual(captures[-1]['model']['kwargs']['num_layers'], 9)
            self.assertEqual(logger_calls[-1]['id'], 'explicit-id')
            self.assertEqual(logger_calls[-1]['resume'], 'never')


if __name__ == '__main__':
    unittest.main()
