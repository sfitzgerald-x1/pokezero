"""Option-A phase profiling tools (gpu_engine.v2e_census, gpu_engine.v2e_phase_profile, scripts/v2e_prof_a1.sh).

CPU only, small: a stand-in step_with_chance whose statements sit on the real phase lines, fake marks, a tiny HLO text,
fake kernel events and a tiny engine table. No real tables, captures, GPU or CUDA build.
"""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from gpu_engine import v2e_census as cz, v2e_phase_profile as pp

ROOT = Path(__file__).resolve().parents[1]
# Statements of a stand-in step_with_chance on the real phase lines (A, B, D, E0, two actions E1/E4, F1, two residual events, G).
CODE = {315: 'def step_with_chance(x, y):', 459: '    a = helper(x) + y', 511: '    b = a * y', 566: '    c = b - x', 670: '    c = c + 1',
    688: '    for order in range(2):', 690: '        c = c * 2 + order', 1600: '        c = jnp.where(c > 0, c, -c)', 1689: '    d = c - y',
    1803: '    for event in range(2):', 1804: '        d = d + event', 1900: '        d = d * 3', 2041: '    return d + a'}
HEADER = ['import jax', 'import jax.numpy as jnp', '', '', '@jax.jit', 'def helper(x):', '    return x * 3 + 1']
LABELS = ['A', 'B', 'D', 'E0', 'E1.0', 'E4.0', 'E1.1', 'E4.1', 'F1', 'F2.0', 'F2.1', 'G']


def stand_in():
    lines = HEADER + [''] * (max(CODE) - len(HEADER))
    for n, text in CODE.items():lines[n - 1] = text
    folder = Path(tempfile.mkdtemp(prefix='v2e-phases-')) / 'gpu_engine';folder.mkdir()
    path = folder / 'transition.py';path.write_text('\n'.join(lines) + '\n')
    spec = importlib.util.spec_from_file_location('v2e_phases_stand_in', path);module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def traced():
    from gpu_engine import v2e_gen
    x = np.arange(4, dtype=np.int32)
    return v2e_gen.trace(stand_in().step_with_chance, (x, x + 2))


class PhaseTableTest(unittest.TestCase):
    def test_phases_tile_step_with_chance(self):
        self.assertEqual(cz.PHASES[0][1], cz.SPAN[0]);self.assertEqual(cz.PHASES[-1][2], cz.SPAN[1])
        for (_, _, last, _), (_, first, _, _) in zip(cz.PHASES, cz.PHASES[1:]):self.assertEqual(first, last + 1)
        self.assertEqual({g for *_, g in cz.PHASES} - {'pre'}, set(cz.GROUPS))

    def test_ranges_match_the_oracle_source(self):
        """Line drift in transition.step_with_chance fails here: its span and every anchor line's text."""
        from gpu_engine import transition
        lines, start = inspect.getsourcelines(transition.step_with_chance)
        self.assertEqual((start, start + len(lines) - 1), cz.SPAN)
        for line, text in cz.ANCHORS.items():
            self.assertTrue(lines[line - start].strip().startswith(text), f'transition.py:{line} is {lines[line - start].strip()[:80]!r}, expected {text!r}')
        for line in (cz.E_LATE, cz.E_RESTART, cz.F2_LOOP, cz.F2_HEAD, cz.F2_BODY):self.assertIsNotNone(cz.phase_of(line))
        self.assertEqual(cz.phase_of(cz.E_LATE), 'E4');self.assertEqual(cz.phase_of(cz.E_RESTART), 'E1')
        self.assertEqual(cz.phase_of(cz.F2_LOOP), 'F2');self.assertEqual(cz.phase_of(cz.F2_HEAD), 'F2')

    def test_boundaries_open_phases(self):
        names = {name for name, *_ in cz.PHASES}
        for tag, label in cz.BOUNDARIES:self.assertTrue(label == 'END' or cz.base(label) in names, tag)

    def test_label_lines(self):
        lines = [None, 460, 511, 540, 600, 680, 700, 900, 1200, 1600, 700, 900, 1650, 1700, 1797, 1804, 1900, 1810, 1835, 1812, 2050, None]
        want = ['outside', 'A', 'B', 'C', 'D', 'E0', 'E1.0', 'E2.0', 'E3.0', 'E4.0', 'E1.1', 'E2.1', 'E4.1', 'F1', 'F2.pre', 'F2.0', 'F2.0',
            'F2.1', 'F2.1', 'F2.2', 'G', 'outside']
        self.assertEqual(cz.label_lines(lines), want)
        self.assertEqual([cz.group_of(l) for l in want[1:5]], ['A', 'B', 'C', 'D'])
        self.assertEqual(cz.group_of('E3.1'), 'E');self.assertEqual(cz.group_of('F2.pre'), 'F');self.assertEqual(cz.group_of('outside'), 'outside')

    def test_listing_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, 'listing.txt');path.write_text('  326:   check\n  459:   rows\n  511:   speed\n 1700:   x\n 2099:   return\nnoise\n')
            self.assertEqual(cz.listing_lines(path), {'pre': 1, 'A': 1, 'B': 1, 'F1': 1, 'G': 1})


class CensusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gpu_engine import v2e_j2c
        cls.closed, cls.avals, _, _ = traced();cls.report = cz.census(cls.closed, v2e_j2c.LANES)

    def test_labels_and_groups(self):
        self.assertEqual(list(self.report['labels']), LABELS)
        self.assertEqual(list(self.report['groups']), ['A', 'B', 'D', 'E', 'F', 'G'])
        self.assertEqual(sum(r['n'] for r in self.report['groups'].values()), self.report['leaf'])
        self.assertEqual(self.report['modules'], {'transition': self.report['leaf']})
        self.assertEqual(self.report['groups']['A']['inline'], self.report['groups']['A']['n'])

    def test_carry_boundaries(self):
        tags = {row['boundary']: row for row in self.report['carry'] if row['boundary']}
        self.assertEqual(set(tags), {'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'})
        self.assertEqual(tags['b8']['label'], 'END');self.assertEqual(tags['b1']['label'], 'B')
        # b4 (start of action 0): c and the inputs a, y are live lane scalars.
        self.assertGreaterEqual(tags['b4']['lane_vars'], 3);self.assertEqual(tags['b4']['lane_elements'], tags['b4']['lane_vars'])

    def test_generated_counts_and_mark_pipeline(self):
        from gpu_engine import v2e_j2c
        with mock.patch.dict(os.environ, {'V2E_CHUNK_LINES': '30'}):
            module = v2e_j2c.generate('step', self.closed, self.avals, profile=True, mode='thread')
        counts = cz.generated_counts(module.marks)
        self.assertLessEqual(set(counts['groups']), set(self.report['groups']));self.assertGreater(counts['groups']['E'], 0)
        chunk = pp.mark_chunks(module);self.assertGreater(chunk.max(), 0)
        labels, modules = pp.mark_table(module.marks);self.assertEqual(list(dict.fromkeys(labels)), LABELS)
        values = np.arange(1, len(chunk) + 1, dtype=np.uint32) * 10   # monotonic offsets: every delta is 10 except chunk starts
        cycles = pp.mark_cycles(values, chunk)
        starts = np.flatnonzero(np.r_[True, chunk[1:] != chunk[:-1]])
        np.testing.assert_array_equal(np.delete(cycles, starts), 10);np.testing.assert_array_equal(cycles[starts], values[starts])
        agg = pp.aggregate(cycles, labels, modules, chunk)
        self.assertEqual(agg['cycles_total'], int(cycles.sum()));self.assertLessEqual(set(agg['by_group']), set(self.report['groups']))
        self.assertEqual(sum(sum(v.values()) for v in agg['by_chunk'].values()), agg['cycles_total'])

    def test_census_markdown(self):
        text = cz.markdown(dict(census=self.report, device_lines={'pre': 2, 'A': 3}, generator=dict(groups={'A': 1})))
        self.assertIn('| A |', text);self.assertIn('| 5 |', text);self.assertIn('b8', text)


class MarkMathTest(unittest.TestCase):
    def test_deltas_restart_per_chunk(self):
        chunk = np.array([0, 0, 0, 1, 1, 2]);values = np.array([5, 9, 20, 3, 10, 7], np.uint32)
        np.testing.assert_array_equal(pp.mark_cycles(values, chunk), [5, 4, 11, 3, 7, 7])

    def test_heaviest_chunks_and_ncu_regex(self):
        report = dict(lanes={'128': dict(by_group={'E': 50, 'F': 30, 'outside': 99}, by_chunk={'0': {'E': 10, 'F': 20}, '1': {'E': 40}, '2': {'F': 10, 'outside': 99}})})
        chunks = pp.heaviest_chunks(report, 128, 2)
        self.assertEqual(chunks, [('E', 1, 40), ('F', 0, 20)])
        self.assertEqual(pp.ncu_regex(chunks), r'regex:(^|::)(chunk_0|chunk_1|chance_kernel|touch_kernel)\(')

    def test_generated_ms_by_chunk_time(self):
        entry = dict(by_chunk={'0': {'E': 30, 'F': 10}, '1': {'F': 10}})
        self.assertEqual(pp.generated_ms(entry, 10., None), {'E': 6., 'F': 4.})
        got = pp.generated_ms(entry, 10., {'0': 100., '1': 300.})   # chunk 0: E 75, F 25; chunk 1: F 300
        self.assertAlmostEqual(got['E'], 10 * 75 / 400);self.assertAlmostEqual(got['F'], 10 * 325 / 400)
        self.assertEqual(pp.generated_ms(entry, 10., {'0': 100.}), {'E': 6., 'F': 4.})   # a chunk without a time: cycle shares
        self.assertIsNone(pp.generated_ms(entry, None, None))


# Generated by jax 0.12 (CPU) for outer -> step_with_chance -> leaf: note each parent_frame_id printed one above.
HLO = '''HloModule jit_outer, is_scheduled=true

FileNames
1 "/x/gpu_engine/transition.py"
2 "/x/gpu_engine/lane_index.py"

FunctionNames
1 "<module>"
2 "outer"
3 "step_with_chance"
4 "leaf"

FileLocations
1 {file_name_id=1 function_name_id=1 line=10 end_line=10 column=4 end_column=35}
2 {file_name_id=1 function_name_id=2 line=9 end_line=9 column=11 end_column=30}
3 {file_name_id=1 function_name_id=3 line=700 end_line=700 column=8 end_column=15}
4 {file_name_id=2 function_name_id=4 line=4 end_line=4 column=11 end_column=23}
5 {file_name_id=1 function_name_id=3 line=1700 end_line=1700 column=11 end_column=16}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=4}
5 {file_location_id=5 parent_frame_id=3}


ENTRY %main.1 (x.1: f32[4]) -> f32[4] {
  %x.1 = f32[4]{0} parameter(0), metadata={op_name="x"}
  %fusion.1 = f32[4]{0} fusion(%x.1), kind=kLoop, calls=%f1, metadata={op_name="jit(chunk)/C_step/sin" stack_frame_id=4}
  %fusion.2 = f32[4]{0} fusion(%fusion.1), kind=kLoop, calls=%f2, metadata={op_name="jit(chunk)/C_step/add" stack_frame_id=5}
  %fusion.3 = f32[4]{0} fusion(%fusion.2), kind=kLoop, calls=%f3, metadata={op_name="jit(chunk)/C_step/add" stack_frame_id=2}
  ROOT %fusion.4 = f32[4]{0} fusion(%fusion.3), kind=kLoop, calls=%f4, metadata={op_name="jit(chunk)/C_forward/dot" stack_frame_id=1}
}
'''


class JaxAttribTest(unittest.TestCase):
    def test_stacks_reach_the_outermost_frame(self):
        stack = pp.hlo_stacks(HLO.splitlines())
        self.assertEqual(stack.offset, 1)
        self.assertEqual(stack(4), [('lane_index.py', 'leaf', 4), ('transition.py', 'step_with_chance', 700), ('transition.py', 'outer', 9),
            ('transition.py', '<module>', 10)])
        self.assertEqual([f[2] for f in stack(5)], [1700, 9, 10])

    def test_index_and_attribution(self):
        index = pp.phase_hlo_index(HLO)
        self.assertEqual((index['fusion.1']['line'], index['fusion.1']['module'], index['fusion.1']['scope']), (700, 'lane_index', 'step'))
        self.assertEqual(index['fusion.2']['line'], 1700);self.assertIsNone(index['fusion.3']['line']);self.assertEqual(index['fusion.4']['scope'], 'forward')
        event = lambda op, dur: dict(name=op, dur=dur, start=0, end=dur, line='Stream #1', stats={'hlo_op': op})
        events = [event('fusion.1', 4_000_000), event('fusion.2', 2_000_000), event('fusion.3', 1_000_000), event('fusion.4', 8_000_000),
            event('unknown.9', 5), dict(name='memcpy', dur=9, start=0, end=9, line='Stream #1', stats={})]
        got = pp.attribute(events, index, steps=2)
        self.assertEqual(got['ms_per_tick_by_group'], {'E': 2.0, 'F': 1.0, 'outside': 0.5})
        self.assertEqual(got['ms_per_tick_by_phase'], {'E1': 2.0, 'F1': 1.0, 'outside': 0.5})
        self.assertEqual(got['ms_per_tick_by_group_category']['E'], {'tracker': 2.0})
        self.assertAlmostEqual(got['step_ms_per_tick'], 3.5);self.assertIn('forward', got['ms_per_tick_by_scope'])

    def test_correct_parent_ids_are_kept(self):
        fixed = HLO.replace('1 {file_location_id=1 parent_frame_id=1}', '1 {file_location_id=1 parent_frame_id=0}')
        for k, p in ((2, 1), (3, 2), (4, 3), (5, 2)):
            fixed = fixed.replace(f'{k} {{file_location_id={k} parent_frame_id={p + 1}}}', f'{k} {{file_location_id={k} parent_frame_id={p}}}')
        stack = pp.hlo_stacks(fixed.splitlines());self.assertEqual(stack.offset, 0)
        self.assertEqual([f[2] for f in stack(4)], [4, 700, 9, 10])


class DivergenceTest(unittest.TestCase):
    def test_union_stats(self):
        stats = pp.UnionStats(warp=4)
        a = np.array([1, 1, 1, 1, 1, 0, 1, 0], bool);b = np.array([0, 0, 0, 0, 0, 1, 0, 1], bool)
        stats.add(dict(E={'x': a, 'y': b}, G={}))
        s = stats.summary(dict(E=.5))['E']
        self.assertEqual(s['warps'], 2);self.assertEqual(s['distinct_classes_per_warp']['mean'], 1.5);self.assertEqual(s['single_class_warps'], .5)
        self.assertEqual(s['lane_features_mean'], 1.);self.assertEqual(s['union_features_mean'], 1.5)
        # F = 2 features; lane cost .5 + .5 * 1/2 = .75; warp cost .5 + .5 * 1.5/2 = .875.
        self.assertAlmostEqual(s['union_factor'], .875 / .75);self.assertAlmostEqual(s['masked_uniform_factor'], 1 / .75)
        self.assertNotIn('G', stats.summary())
        self.assertAlmostEqual(pp.weighted_factor(dict(E=dict(union_factor=2.), F=dict(union_factor=1.)), dict(E=3, F=1)), 1.75)

    def test_lane_features_on_a_tiny_table(self):
        import jax
        import jax.numpy as j
        from gpu_engine import transition as tr
        from gpu_engine.state import empty_state, usable_moves
        types = ['_none', 'normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
            'rock', 'ghost', 'dragon', 'dark', 'steel']
        md = {'vocabulary': dict(moves=['_none', 'struggle', 'tackle', 'pursuit'], abilities=['_none', 'intimidate'], items=['_none', 'leftovers'],
                species=['_none', 'pikachu'], types=types),
            'moves': [None, dict(id='struggle', base_power=50, type=1, source_type='Normal', category='Physical', pp=1, accuracy=True, priority=0, flags={}),
                dict(id='tackle', base_power=35, type=1, source_type='Normal', category='Physical', pp=35, accuracy=95, priority=0, flags={'contact': 1}),
                dict(id='pursuit', base_power=40, type=16, source_type='Dark', category='Physical', pp=20, accuracy=100, priority=0, flags={'contact': 1})],
            'type_chart': np.ones((17, 17)).tolist()}
        table = tr.tables_from_metadata(md);b = 4
        st = empty_state(b)._replace(species=j.ones((b, 2, 6), j.int32), hp=j.full((b, 2, 6), 100, j.int32), max_hp=j.full((b, 2, 6), 100, j.int32),
            level=j.full((b, 2, 6), 80, j.int32), done=j.array([False, False, False, True]), moves=j.zeros((b, 2, 6, 4), j.int32).at[..., 0].set(2).at[:, 0, 0, 1].set(3),
            pp=j.full((b, 2, 6, 4), 10, j.int32), item=j.ones((b, 2, 6), j.int32), ability=j.ones((b, 2, 6), j.int32), status=j.zeros((b, 2, 6), j.int32).at[1, 0, 0].set(2))
        actions = np.array([[0, 0], [1, 4], [0, 4], [0, 0]], np.int32)   # lane 1: Pursuit against a switch
        columns = pp.table_columns(table)
        self.assertIn('.pursuit.pursuit', columns['move']);self.assertIn('.entry.intimidate', columns['ability']);self.assertIn('.items.leftovers', columns['item'])
        mask = np.asarray(tr.action_mask(st, table));usable = np.asarray(usable_moves(st))
        f = pp.lane_features(jax.device_get(st), actions, jax.device_get(table), columns, mask, usable)
        self.assertEqual(set(f), set(cz.GROUPS))
        np.testing.assert_array_equal(f['A']['kind:done'], [False, False, False, True])
        np.testing.assert_array_equal(f['A']['kind:ordinary'], [True, True, True, False])
        np.testing.assert_array_equal(f['C']['pursuit'], [False, True, False, False])
        np.testing.assert_array_equal(f['D']['switch'], [False, True, True, False])
        np.testing.assert_array_equal(f['D']['enter:.entry.intimidate'], [False, True, True, False])
        np.testing.assert_array_equal(f['E']['move:.pursuit.pursuit'], [False, True, False, False])
        np.testing.assert_array_equal(f['E']['status:2'], [False, True, False, False])
        np.testing.assert_array_equal(f['F']['item:.items.leftovers'], [True, True, True, False])
        stats = pp.UnionStats(warp=2);stats.add(f);self.assertEqual(set(stats.summary()), set(cz.GROUPS))


class ReportTest(unittest.TestCase):
    def test_decision_rule(self):
        d = pp.decide({'A': 1., 'D': 1., 'E': 5., 'F': 2., 'G': 1.}, {})
        self.assertEqual(d['M2'], 'F + G');self.assertEqual(d['M4'], 'E split (E1-E2 | E3-E4)')
        d = pp.decide({'A': 1., 'D': 3., 'E': 5., 'F': 0.5, 'G': .5}, {})
        self.assertEqual(d['M2'], 'D');self.assertEqual(d['M4'], 'E split (E1-E2 | E3-E4)')
        d = pp.decide(None, {'E': dict(n=3), 'F': dict(n=4), 'G': dict(n=3)})
        self.assertEqual((d['M2'], d['M4'], d['basis']), ('F + G', 'E whole', 'census equations (no timing)'))

    def test_split_shares(self):
        s = pp.split_shares(1.0, .6, .1)
        self.assertAlmostEqual(s['forward_ms'], 400.);self.assertAlmostEqual(s['engine_ms'], 500.);self.assertAlmostEqual(s['depth1_ms'], 100.)
        self.assertAlmostEqual(s['forward_share'], .4);self.assertAlmostEqual(s['engine_share'], .5);self.assertEqual(s['total_ms'], 1000.)
        self.assertIsNone(pp.split_shares(1.0, None, .1)['forward_ms'])
        args = pp.parser().parse_args(['pricer-split', '--d3-root', '/d', '--chunk', '0', '--out', '/o'])
        ns = pp.pricer_args(args);self.assertEqual((ns.capacity, ns.w_values, ns.block_lanes), (8, (2,), 2187))

    def test_ncu_csv(self):
        with tempfile.TemporaryDirectory() as d:
            details, raw = Path(d, 'details.csv'), Path(d, 'raw.csv')
            details.write_text('"ID","Kernel Name","Section Name","Metric Name","Metric Unit","Metric Value"\n'
                '"0","v2e_gen_step::chunk_7(v2e_gen_step::Params)","Warp State Statistics","Warp Cycles Per Issued Instruction","cycle","13.30"\n'
                '"0","v2e_gen_step::chunk_7(v2e_gen_step::Params)","Memory Workload Analysis","L1/TEX Hit Rate","%","63.1"\n'
                '"0","v2e_gen_step::chunk_7(v2e_gen_step::Params)","Memory Workload Analysis","Mem Busy","%","4"\n'
                '"1","chance_kernel(long, unsigned int const*)","Memory Workload Analysis","L2 Hit Rate","%","1,40.5"\n')
            raw.write_text('"ID","Kernel Name","smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",'
                '"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"\n"","","ratio","ratio"\n'
                '"0","v2e_gen_step::chunk_7(v2e_gen_step::Params)","5.6","4.5"\n')
            got = pp.ncu_summary(details, raw)
            self.assertEqual(got['0']['chunk'], 7);self.assertEqual(got['0']['metrics']['Warp Cycles Per Issued Instruction']['value'], 13.3)
            self.assertNotIn('Mem Busy', got['0']['metrics']);self.assertEqual(list(got['0']['stalls']), ['no_instruction', 'long_scoreboard'])
            self.assertIsNone(got['1']['chunk']);self.assertEqual(got['1']['metrics']['L2 Hit Rate']['value'], 140.5)
            marks = dict(lanes={'128': dict(by_chunk={'7': {'E': 5, 'F': 9}})})
            self.assertEqual(pp.ncu_by_group(got, marks)['0']['group'], 'F')
        self.assertEqual(pp.CHUNK.search('_ZN12v2e_gen_step8chunk_12ENS_6ParamsE').group(1), '12')
        self.assertEqual(pp.CHUNK.search('v2e_gen_step::chunk_120(v2e_gen_step::Params)').group(1), '120')

    def test_nsys_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, 'walls_cuda_gpu_trace.csv')
            path.write_text('Start (ns),Duration (ns),GrdX,GrdY,Name\n1,10,4,1,v2e_gen_step::chunk_0(v2e_gen_step::Params)\n'
                '2,30,4,1,v2e_gen_step::chunk_0(v2e_gen_step::Params)\n3,5,64,1,v2e_gen_step::chunk_1(v2e_gen_step::Params)\n4,7,1,1,memset\n')
            self.assertEqual(pp.nsys_chunks(path), {128: {'0': 20.}, 2048: {'1': 5.}})

    def test_report_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d);write = lambda name, file, data: (d / name).mkdir() or (d / name / file).write_text(json.dumps(data))
            groups = {g: dict(n=n, tracker=n // 2, hook_share=.5) for g, n in zip(cz.GROUPS, (10, 1, 5, 20, 40, 30, 4))}
            groups['outside'] = dict(n=2, tracker=0, hook_share=0.)
            write('census', 'census.json', dict(census=dict(groups=groups), device_lines={'pre': 9, 'A': 24, 'E1': 98, 'E3': 140}))
            chunk = lambda **v: v
            write('marks', 'marks.json', dict(lanes={str(l): dict(by_chunk={'0': chunk(A=10, E=30), '1': chunk(F=40, G=10, outside=10)},
                by_group_category={'E': dict(tracker=27, inline=3)}) for l in (128, 2048)}))
            write('walls', 'walls.json', dict(lanes={'2048': dict(generated=dict(ms_median=10.), jax=dict(ms_median=9.),
                generated_trace=dict(chunk_ns_median={'0': 400., '1': 600.}))}))
            write('jax', 'jax_attrib.json', dict(lanes={'2048': dict(ms_per_tick_by_group={'E': 3., 'F': 2.})}))
            write('divergence', 'divergence.json', dict(groups={'E': dict(union_factor=1.4)}, weighted_union_factor=1.2))
            write('pricer', 'pricer_split.json', dict(split=pp.split_shares(.3, .2, .05)))
            argv = ['report', '--out', str(d / 'report')] + sum((['--' + k, str(d / k)] for k in ('census', 'marks', 'walls', 'jax', 'divergence', 'pricer')), [])
            with mock.patch('builtins.print'):self.assertEqual(pp.main(argv), 0)
            report = json.loads((d / 'report' / 'phase_attribution.json').read_text());rows = {r['phase']: r for r in report['rows']}
            self.assertEqual(rows['A']['device_lines'], 33);self.assertEqual(rows['E']['device_lines'], 238)
            self.assertAlmostEqual(rows['E']['generated_ms_2048'], 10 * 300 / 1000);self.assertAlmostEqual(rows['F']['generated_ms_2048'], 10 * 400 / 1000)
            self.assertIsNone(rows['E']['generated_ms_128']);self.assertEqual(rows['E']['jax_ms_2048'], 3.);self.assertEqual(rows['E']['union_factor'], 1.4)
            self.assertAlmostEqual(rows['E']['hook_share_cycles_2048'], .9);self.assertIn('outside', rows)
            self.assertEqual(report['decision']['M2'], 'F + G');self.assertEqual(report['decision']['M4'], 'E whole')
            text = (d / 'report' / 'phase_attribution.md').read_text()
            self.assertIn('Live pricer split', text);self.assertIn('M2 = F + G', text)


class CommandLineTest(unittest.TestCase):
    def test_parsers(self):
        p = pp.parser()
        run = ['--config', 'c', '--reference-run', 'r', '--shards', 's', '--out', 'o']
        self.assertEqual(p.parse_args(['marks', *run]).lanes, [128, 2048])
        self.assertEqual(p.parse_args(['walls', *run]).lanes, [128, 2048, 8192])
        self.assertEqual(p.parse_args(['ncu-driver', *run]).lanes, 128)
        self.assertEqual(p.parse_args(['divergence', *run]).limit, 100)
        self.assertEqual(p.parse_args(['jax-attrib', '--profile-dir', 'x', '--out', 'o']).lanes, [128, 2048])
        self.assertEqual(cz.parser().parse_args(['--config', 'c', '--reference-run', 'r', '--out', 'o']).games, 1)
        with self.assertRaises(SystemExit), mock.patch('sys.stderr'):pp.main(['pricer-split', '--d3-root', '/d', '--chunk', '0', '--out', '/o'])

    def test_job_script_dry_run(self):
        result = subprocess.run(['bash', str(ROOT / 'scripts' / 'v2e_prof_a1.sh'), '--config', '/c.json', '--reference-run', '/r', '--shards', '/s/shards',
            '--out', '/o/prof', '--build-dir', '/b', '--python', '/py', '--d3-root', '/d3', '--chunk', '0', '--champion-archive', '/w.npz',
            '--champion-archive-sha256', 'a', '--champion-checkpoint-sha256', 'b', '--dry-run'], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        steps = [line.split()[1] for line in result.stdout.splitlines() if line.startswith('=== ')]
        self.assertEqual(steps, ['census', 'marks', 'walls', 'nsys', 'nsys-stats', 'jax-profile', 'jax-attrib', 'ncu', 'ncu-export', 'divergence', 'pricer', 'report'])
        commands = [line for line in result.stdout.splitlines() if line.startswith('+ ')]
        self.assertTrue(all('timeout --kill-after=60' in c for c in commands))
        self.assertTrue(any('nsys profile' in c for c in commands));self.assertTrue(any('ncu --set full' in c for c in commands))
        self.assertTrue(any('--frames 40 --save-trace --components' in c for c in commands))
        self.assertTrue(any('JAX_DEFAULT_MATMUL_PRECISION=highest' in c and 'pricer-split' in c for c in commands))
        missing = subprocess.run(['bash', str(ROOT / 'scripts' / 'v2e_prof_a1.sh'), '--dry-run'], capture_output=True, text=True, timeout=60)
        self.assertEqual(missing.returncode, 2)


if __name__ == '__main__':
    unittest.main()
