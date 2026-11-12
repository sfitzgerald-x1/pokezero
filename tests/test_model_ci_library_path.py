"""Exercise the workflow's loader assignment against real Linux ELF libraries.

The model seam needs both Torch and the setup-python runtime libraries. This
test reproduces the exit-127 failure without building Torch or the Rust crate.
"""
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(platform.system() == 'Linux' and shutil.which('cc'), 'Linux ELF loader and C compiler required')
class ModelRuntimePathTests(unittest.TestCase):
    def test_model_step_preserves_python_and_torch_library_directories(self):
        workflow = Path(__file__).resolve().parents[1] / '.github/workflows/engine-fidelity-gates.yml'
        step = workflow.read_text().split('      - name: Model-feature abort seam\n', 1)[1].split('\n      - name:', 1)[0]
        assignment, = [line.strip().removesuffix('\\').rstrip() for line in step.splitlines()
                       if line.strip().startswith('LD_LIBRARY_PATH=')]
        with tempfile.TemporaryDirectory(prefix='ci-loader-') as tmp:
            root = Path(tmp)
            python_lib = root / 'python runtime'
            torch_root = root / 'fake torch'
            torch_lib = torch_root / 'lib'
            python_lib.mkdir();torch_lib.mkdir(parents=True)
            # import torch locates this deliberately tiny stand-in. Its only
            # role is to exercise the exact workflow path discovery expression.
            (torch_root / 'torch.py').write_text('')
            for name, value, directory in (('ci_python_probe', 20, python_lib), ('ci_torch_probe', 22, torch_lib)):
                source = root / (name + '.c')
                source.write_text(f'int {name}(void) {{ return {value}; }}\n')
                subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(directory / ('lib' + name + '.so'))], check=True)
            source = root / 'main.c';binary = root / 'probe'
            source.write_text('#include <stdio.h>\nint ci_python_probe(void); int ci_torch_probe(void);\n'
                              'int main(void) { printf("%d\\n", ci_python_probe()+ci_torch_probe()); return 0; }\n')
            subprocess.run(['cc', str(source), '-L' + str(python_lib), '-L' + str(torch_lib),
                            '-lci_python_probe', '-lci_torch_probe', '-o', str(binary)], check=True)
            bin_dir = root / 'bin';bin_dir.mkdir()
            python = bin_dir / 'python'
            python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n');python.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
                       PYTHONPATH=str(torch_root), LD_LIBRARY_PATH=str(python_lib) +
                       (os.pathsep + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
            def run(expression):
                return subprocess.run(['bash', '-eu', '-c', expression + ' ' + shlex.quote(str(binary))],
                                      env=env, text=True, capture_output=True)
            # Prove the fixture actually detects the historical failure (the step without its inherited-path suffix).
            suffix = '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'
            self.assertIn(suffix, assignment, 'the step must append the inherited LD_LIBRARY_PATH')
            old = run(assignment.replace(suffix, ''))
            self.assertEqual(old.returncode, 127, old.stderr)
            self.assertIn('libci_python_probe.so', old.stderr)
            fixed = run(assignment)
            self.assertEqual(fixed.returncode, 0, fixed.stderr)
            self.assertEqual(fixed.stdout.strip(), '42')
