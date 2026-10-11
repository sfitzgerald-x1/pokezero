import ast
import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from pokezero.mcts_eval.paper_reference import ReferenceRefusal


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'wider_search_comparison.py'
spec = importlib.util.spec_from_file_location('wider_conditioning_options_under_test', SCRIPT)
wider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wider)


class RegisteredConditioningTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.bindings = 0

    def bind_options(self, registration):
        self.bindings += 1
        path = self.directory / f'options-{self.bindings}.json'
        raw = json.dumps(registration['reference']['conditioning_options']).encode()
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        registration['reference']['conditioning_options_binding'] = dict(path=str(path), sha256=digest)
        registration['input_hashes'] = {str(path): digest}
        return registration

    def candidate(self):
        return self.bind_options(dict(reference=dict(staged_substitute_conditioning=True, conditioning_batch_size=8,
            workers=20, exchange_trajectories=10, initial_dispatch_workers=6,
            conditioning_options=dict(history_particles=32, guide_history_actions=True,
                history_chance_pool=1, batch_history_chance=False, guide_history_chance=True,
                guide_history_accuracy=True, membership_first=True, native_membership_batch_size=8,
                public_anchor_constraints=True, guide_anchor_genders=True, early_encore_potential=True,
                collect_phase_timing=True, history_prefix_reuse=True, history_prefix_public_refresh=True))))

    def test_historical_defaults_preserved(self):
        result = wider.registered_reference_configuration({})
        self.assertEqual(result, dict(workers=20, batch_size=10, initial_dispatch_workers=20,
            factory_options=dict(allow_earlier_compatible_template=True, max_known_set_draws=128,
                staged_substitute_conditioning=False, conditioning_batch_size=1)))

    def test_candidate_passes_every_option_without_mutating_registration(self):
        registration = self.candidate()
        before = copy.deepcopy(registration)
        configuration = wider.registered_reference_configuration(registration)
        self.assertEqual(configuration['initial_dispatch_workers'], 6)
        for key,value in registration['reference']['conditioning_options'].items():
            self.assertEqual(configuration['factory_options'][key], value)
        self.assertEqual(before, registration)

    def test_reject_unknown_identity_type_or_conflicting_options(self):
        for key,value in [('checkpoint', 'foreign'), ('unknown', True),
                ('guide_history_accuracy', 1), ('history_particles', True),
                ('allow_earlier_compatible_template', 1), ('max_known_set_draws', 129),
                ('staged_substitute_conditioning', False), ('conditioning_batch_size', 1)]:
            registration = self.candidate()
            registration['reference']['conditioning_options'][key] = value
            self.bind_options(registration)
            with self.assertRaises((RuntimeError, ReferenceRefusal)):
                wider.registered_reference_configuration(registration)

    def test_runtime_relationship_guards_are_not_bypassed(self):
        for key,value in [('guide_history_chance', False), ('history_particles', 0),
                ('history_prefix_reuse', False), ('batch_history_chance', True),
                ('public_anchor_constraints', False), ('membership_first', False)]:
            registration = self.candidate()
            registration['reference']['conditioning_options'][key] = value
            self.bind_options(registration)
            with self.assertRaises((RuntimeError, ReferenceRefusal)):
                wider.registered_reference_configuration(registration)

    def test_particles_cannot_skip_staged_decoder_binding_and_disclosure(self):
        registration = self.candidate()
        registration['reference'].update(staged_substitute_conditioning=False, conditioning_batch_size=1)
        with self.assertRaisesRegex(RuntimeError, 'explicit staged qualification and decoder binding'):
            wider.registered_reference_configuration(registration)

    def test_worker_dispatch_shape_cannot_change_silently(self):
        for key,value in [('workers', 19), ('workers', 20.), ('exchange_trajectories', 9),
                ('initial_dispatch_workers', 0), ('initial_dispatch_workers', 21),
                ('initial_dispatch_workers', True)]:
            registration = self.candidate()
            registration['reference'][key] = value
            with self.assertRaises(RuntimeError):
                wider.registered_reference_configuration(registration)

    def test_changed_dispatch_cannot_register_strength_without_fresh_qualification(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh full-game qualification first'):
            wider.register(SimpleNamespace(staged_substitute_conditioning=False,
                conditioning_batch_size=1, initial_dispatch_workers=6, qualification=False))

    def test_new_candidate_cannot_use_differently_configured_qualification(self):
        configuration = self.candidate()
        wider.validate_candidate_configuration_pair(configuration, copy.deepcopy(configuration))
        for key,value in [('guide_history_accuracy', False), ('history_prefix_public_refresh', False)]:
            old = copy.deepcopy(configuration)
            old['reference']['conditioning_options'][key] = value
            self.bind_options(old)
            with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
                wider.validate_candidate_configuration_pair(old, configuration)
        with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
            wider.validate_candidate_configuration_pair({}, configuration)
        old = copy.deepcopy(configuration)
        old['reference']['initial_dispatch_workers'] = 20
        with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
            wider.validate_candidate_configuration_pair(old, configuration)
        # Existing explicit historical transfer validators retain responsibility
        # for legacy staged-kernel amendments, not for the new candidate flags.
        wider.validate_candidate_configuration_pair({}, {'reference': {
            'staged_substitute_conditioning': True, 'conditioning_batch_size': 8}})

    def test_declared_base_settings_must_match_actual_constructor(self):
        for key,value in [('max_known_set_draws', 129), ('allow_earlier_compatible_template', False),
                ('allow_earlier_compatible_template', 1)]:
            registration = self.candidate()
            registration['reference'][key] = value
            with self.assertRaisesRegex(RuntimeError, 'base settings drift'):
                wider.registered_reference_configuration(registration)

    def test_raw_binding_and_decoded_options_cannot_diverge(self):
        registration = self.candidate()
        del registration['reference']['conditioning_options_binding']
        with self.assertRaisesRegex(RuntimeError, 'raw-file input binding'):
            wider.registered_reference_configuration(registration)
        registration = self.candidate()
        registration['reference']['conditioning_options']['guide_history_accuracy'] = False
        with self.assertRaisesRegex(RuntimeError, 'raw and parsed'):
            wider.registered_reference_configuration(registration)
        registration = self.candidate()
        binding = registration['reference']['conditioning_options_binding']
        Path(binding['path']).write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'raw and parsed'):
            wider.registered_reference_configuration(registration)
        registration = self.candidate()
        registration['input_hashes'] = {}
        with self.assertRaisesRegex(RuntimeError, 'raw-file input binding'):
            wider.registered_reference_configuration(registration)

    def test_real_driver_consumes_bound_factory_and_dispatch_fields(self):
        tree = ast.parse(SCRIPT.read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        factory = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'ShowdownWorkerFactory')
        pool = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'ParallelTrajectorySearch')
        self.assertEqual(ast.unparse(factory.keywords[0].value), "reference_configuration['factory_options']")
        self.assertIsNone(factory.keywords[0].arg)
        self.assertEqual({keyword.arg: ast.unparse(keyword.value) for keyword in pool.keywords}, {
            'workers': "reference_configuration['workers']",
            'batch_size': "reference_configuration['batch_size']",
            'initial_dispatch_workers': "reference_configuration['initial_dispatch_workers']"})


if __name__ == '__main__':
    unittest.main()
