import copy
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import tempfile
import unittest
from unittest.mock import patch

from scripts import compare_projected_v4_encoder as cli


def result():
    from pokezero.recorded_player_encoder_runner import VISIBILITY
    from pokezero.saved_player_projection import (
        PROJECTION_SCOPE, STRUGGLE_WARNING_POLICY, PINNED_SHOWDOWN_REVISION)
    identity = dict(visibility=VISIBILITY, scope=PROJECTION_SCOPE,
        struggle_warning_policy=STRUGGLE_WARNING_POLICY,
        showdown_revision=PINNED_SHOWDOWN_REVISION,
        public_wire_qualified=False, actual_captured_wire=False, captured_player_wire=False,
        owning_stream_bytes_reconstructed=False, original_reference_values_compared=False,
        fullencoder=False, full_encoder_admitted=False)
    metrics = dict(identity,
        schema='pokezero.B2-projected-player-comparison-metrics.v1',
        status='PROJECTED_PLAYER_METRICS_ONLY_REQUIRES_RUNTIME_ATTESTATION',
        views=12, expected_views=12, complete_view_count=True,
        model_forward_complete=True, model_forward_views=12,
        projected_input_numeric_criteria_met=False, projected_forward_numeric_criteria_met=False,
        projected_comparison_numeric_criteria_met=False)
    return dict(identity, metrics=metrics,
        schema='pokezero.B2-projected-player-collection-comparison.v1',
        status='COMPLETE_FIXED_PROJECTED_PLAYER_METRICS_ONLY', new_games=0, simulator_steps=0)


class ProjectedCliTests(unittest.TestCase):
    def args(self, root, mode='paired-forward'):
        return ['--checkpoint', 'unopened-checkpoint', '--scratch-checkpoint', str(root/'private.pt'),
            '--showdown', 'unopened-source', '--node', 'unopened-node', '--collection', 'unopened-corpus',
            '--collection-sha256', 'a'*64, '--cohort-sha256', 'b'*64,
            '--output', str(root/'report'), '--threads', '4', '--mode', mode]

    def test_complete_negative_result_keeps_false_qualification_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = object()
            adapters = {seat: object() for seat in ('p1', 'p2')}
            received = []
            module = ModuleType('pokezero.projected_encoder_collection')
            def compare(collection, **kwargs):
                received.append((collection, kwargs))
                return result()
            module.compare_projected_collection = compare
            with (patch.object(cli, 'build_runtime', return_value=(adapters, config, {'fake': True})) as build,
                    patch.dict('sys.modules', {'pokezero.projected_encoder_collection': module}),
                    patch('builtins.print')):
                self.assertEqual(cli.main(self.args(root)), 0)
            self.assertEqual(build.call_count, 1)
            self.assertTrue(build.call_args.args[0].forward)
            self.assertIs(received[0][1]['adapters'], adapters)
            self.assertIs(received[0][1]['reference_config'], config)
            self.assertTrue(received[0][1]['with_forward'])
            final = json.loads((root/'report/comparison.json').read_text())
            self.assertFalse(final['metrics']['projected_forward_numeric_criteria_met'])
            attempt = json.loads((root/'report/attempt.json').read_text())
            self.assertEqual(attempt['actual_exit'], 0)
            self.assertFalse(attempt['actual_captured_wire'] or attempt['full_encoder_admitted'])

    def test_partial_forward_incomplete_count_and_qualification_overclaims_refuse(self):
        for change in ('views', 'forward', 'wire', 'full', 'games', 'bool_count'):
            value = result()
            if change == 'views': value['metrics']['views'] = 11
            elif change == 'forward': value['metrics']['model_forward_views'] = 11
            elif change == 'wire': value['actual_captured_wire'] = True
            elif change == 'full': value['metrics']['full_encoder_admitted'] = True
            elif change == 'games': value['new_games'] = 1
            else: value['metrics'].update(views=True, expected_views=True)
            with self.subTest(change=change), self.assertRaises(ValueError):
                cli.validate_result(value, with_forward=True)

    def test_report_identity_and_all_qualification_aliases_refuse_before_publication(self):
        cases = []
        for container in ('result', 'metrics'):
            for key in ('schema', 'status', 'visibility', 'scope', 'struggle_warning_policy',
                        'showdown_revision'):
                for change in ('missing', 'different'):
                    value = result()
                    payload = value if container == 'result' else value['metrics']
                    if change == 'missing':
                        del payload[key]
                    else:
                        payload[key] = 'trusted-or-other-contract'
                    cases.append((f'{container}.{key}.{change}', value))
            for key in ('public_wire_qualified', 'actual_captured_wire', 'captured_player_wire',
                        'owning_stream_bytes_reconstructed', 'original_reference_values_compared',
                        'fullencoder', 'full_encoder_admitted'):
                for change in ('missing', 'true', 'zero'):
                    value = result()
                    payload = value if container == 'result' else value['metrics']
                    if change == 'missing':
                        del payload[key]
                    else:
                        payload[key] = True if change == 'true' else 0
                    cases.append((f'{container}.{key}.{change}', value))
        trusted = result()
        for key in tuple(trusted['metrics']):
            if key.startswith('projected_'):
                trusted['metrics'][key.replace('projected_', 'recorded_', 1)] = trusted['metrics'].pop(key)
        cases.append(('trusted-metrics-substitution', trusted))
        for name, value in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                module = ModuleType('pokezero.projected_encoder_collection')
                module.compare_projected_collection = lambda *args, **kwargs: value
                with (patch.object(cli, 'build_runtime', return_value=({}, object(), {})),
                        patch.dict('sys.modules', {'pokezero.projected_encoder_collection': module}),
                        patch('builtins.print')):
                    self.assertEqual(cli.main(self.args(root)), 1)
                self.assertFalse((root/'report/comparison.json').exists())
                attempt = json.loads((root/'report/attempt.json').read_text())
                self.assertEqual(attempt['status'], 'FAILED')
                self.assertEqual(attempt['actual_exit'], 1)

    def test_late_collection_and_publication_errors_never_publish_success(self):
        for failure in ('collection', 'publication'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                module = ModuleType('pokezero.projected_encoder_collection')
                module.compare_projected_collection = lambda *a, **kw: result()
                if failure == 'collection':
                    def compare(*args, **kwargs):
                        raise ValueError('late owned history error')
                    module.compare_projected_collection = compare
                publish = cli.publish_comparison
                def publish_bad(*args):
                    raise OSError('report publication failed')
                with (patch.object(cli, 'build_runtime', return_value=({}, object(), {})),
                        patch.dict('sys.modules', {'pokezero.projected_encoder_collection': module}),
                        patch.object(cli, 'publish_comparison', side_effect=publish_bad
                                     if failure == 'publication' else publish), patch('builtins.print')):
                    self.assertEqual(cli.main(self.args(root)), 1)
                self.assertFalse((root/'report/comparison.json').exists())
                self.assertEqual(json.loads((root/'report/attempt.json').read_text())['status'], 'FAILED')

    def test_single_runtime_load_returns_own_reference_config_and_two_adapters(self):
        from pokezero.recorded_mirror_roots import CHAMPION_CONTRACT
        contract = copy.deepcopy(CHAMPION_CONTRACT)
        model, vocab, config, catalog = object(), object(), object(), object()
        calls = []
        class Adapter:
            def __init__(self, got, **kwargs):
                self.catalog, self.kwargs = got, kwargs
                calls.append(self)
        torch = SimpleNamespace(__version__='fake-cpu', backends=SimpleNamespace(
            mha=SimpleNamespace(set_fastpath_enabled=lambda value: None)),
            set_float32_matmul_precision=lambda value: None,
            get_float32_matmul_precision=lambda: 'highest')
        jax = SimpleNamespace(__version__='fake-cpu', config=SimpleNamespace(update=lambda *args: None),
            devices=lambda: [SimpleNamespace(platform='cpu')])
        gpu = ModuleType('gpu_engine')
        gpu.public_belief = SimpleNamespace(compile_source=lambda source: source)
        gpu.v4_encoder = SimpleNamespace(compile_encoder=lambda *args, **kwargs: catalog)
        loads = []
        def load(*args):
            loads.append(args)
            return model, object(), vocab, config, contract
        modules = {'pokezero.mirror_value_probe': SimpleNamespace(load_runtime=load, CHAMPION_SHA='c'*64),
            'pokezero.neural_policy': SimpleNamespace(require_torch=lambda: torch),
            'pokezero.dex': SimpleNamespace(load_showdown_dex_cached=lambda root: object()),
            'pokezero.randbat': SimpleNamespace(load_gen3_randbat_source_cached=lambda root:
                SimpleNamespace(metadata=SimpleNamespace(source_hash=contract['belief_set_source_hash']))),
            'pokezero.recorded_encoder_runner': SimpleNamespace(JaxRecordedAdapter=Adapter),
            'jax': jax, 'gpu_engine': gpu}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root/'node';node.write_text('not executed');node.chmod(0o555)
            args = SimpleNamespace(node=str(node), checkpoint='not opened', scratch_checkpoint=str(root/'private'),
                showdown='not executed', threads=4, forward=True)
            with (patch.dict('sys.modules', modules), patch.object(cli, 'snapshot_checkpoint', return_value={'fake': True}),
                    patch.dict('os.environ', {}, clear=True)):
                adapters, actual_config, receipt = cli.build_runtime(args)
            self.assertEqual(len(loads), 1)
            self.assertIs(actual_config, config)
            self.assertIsNot(adapters['p1'], adapters['p2'])
            self.assertTrue(all(adapter.catalog is catalog for adapter in calls))
            self.assertTrue(all(adapter.kwargs['model'] is model for adapter in calls))
            self.assertEqual(receipt['independent_player_adapters'], 2)
