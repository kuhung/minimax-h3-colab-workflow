"""Offline checks for generated notebook loading, without GPU or network."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('builder', Path(__file__).resolve().parents[1] / 'scripts/build_v3_notebooks.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

class LoadingTests(unittest.TestCase):
    def test_loading_paths(self):
        tree = ast.parse(builder.get_cell_models('i2v'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'check_safetensors_file', 'ensure_weight_fast'}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(Path=Path, struct=struct, json=json, shutil=shutil,
                       LOCAL_MODELS=root/'local', DRIVE_MODELS=root/'drive',
                       ALLOW_DRIVE_FALLBACK=True, BACKUP_WEIGHTS_TO_DRIVE=False, log=lambda _: None)
            exec(compile(ast.Module(body=functions, type_ignores=[]), '<loader>', 'exec'), env)
            header = json.dumps({'x': {'data_offsets': [0, 1]}}).encode()
            payload = struct.pack('<Q', len(header)) + header + b'x'
            calls = []
            def download(**kwargs):
                calls.append('hf')
                p = root/'download'; p.write_bytes(payload); return str(p)
            def copy(src, dst, desc):
                calls.append('drive'); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
            env.update(hf_hub_download=download, copy_with_progress=copy)
            load = env['ensure_weight_fast']
            p = load('vae/test.safetensors', 'vae', len(payload))
            self.assertEqual(calls, ['hf'])
            load('vae/test.safetensors', 'vae', len(payload))
            self.assertEqual(calls, ['hf'])
            p.write_bytes(b'broken')
            drive = env['DRIVE_MODELS']/'vae/test.safetensors'; drive.parent.mkdir(parents=True); drive.write_bytes(payload)
            def fail(**kwargs): raise OSError('offline')
            env['hf_hub_download'] = fail
            load('vae/test.safetensors', 'vae', len(payload))
            self.assertEqual(calls, ['hf', 'drive'])
            p.unlink(); drive.write_bytes(b'broken')
            with self.assertRaises(RuntimeError): load('vae/test.safetensors', 'vae', len(payload))

if __name__ == '__main__': unittest.main()
