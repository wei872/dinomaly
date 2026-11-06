import argparse
import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from fabric_config import apply_config_defaults


class ConfigTests(unittest.TestCase):
    def parser(self):
        # Build the actual CLI declarations without importing torch/torchvision.
        module = ast.parse(Path('fabric_pipeline.py').read_text(encoding='utf-8'))
        main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        prefix = []
        for node in main.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, 'id', '') == 'apply_config_defaults':
                break
            prefix.append(node)
        namespace = dict(argparse=argparse, Path=Path, __file__=str(Path('fabric_pipeline.py').resolve()),
                         __doc__='test', torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
        exec(compile(ast.Module(body=prefix, type_ignores=[]), '<CLI>', 'exec'), namespace)
        return namespace['parser']

    def test_root_config_and_dataset_classes(self):
        parser = self.parser()
        argv = ['train']
        apply_config_defaults(parser, argv)
        args = parser.parse_args(argv)
        self.assertEqual(args.size, 448)
        self.assertEqual(args.feature_batch_size, 8)
        with open(args.data, encoding='utf-8') as stream:
            spec = yaml.safe_load(stream)
        self.assertEqual(spec['nc'], 19)
        self.assertEqual(len(spec['names']), 19)

    def test_command_line_overrides_yaml(self):
        parser = self.parser()
        argv = ['train', '--size', '320', '--feature-batch-size', '4']
        apply_config_defaults(parser, argv)
        args = parser.parse_args(argv)
        self.assertEqual(args.size, 320)
        self.assertEqual(args.feature_batch_size, 4)

    def test_rejects_legacy_config(self):
        parser = self.parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            apply_config_defaults(parser, ['train', '--config', 'config_dino_legacy.yaml'])


if __name__ == '__main__':
    unittest.main()
