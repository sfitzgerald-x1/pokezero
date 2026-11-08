import copy
import hashlib
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pokezero import depthn_live_panel as panel
from scripts import depthn_extension_dispatch as E
from scripts import depthn_screen_amended_verdict as V
from tests import test_depthn_live_panel as T


def setUpModule(): T.setUpModule()   # the panel suite's module-wide patches (test declarations are not tracked files)


def tearDownModule(): T.tearDownModule()


def files_for(run):
    """The committed amendment chain, re-bound to a fixture run's declaration and tree: {name: (value, sha256)}."""
    real = E.committed();out = {};shas = {}
    for name in ('amendment', 'addendum', 'both', 'extension'):
        value = copy.deepcopy(real[name][0]);value['declaration_sha256'] = run['declaration_sha256']
        value['run'] = dict(value['run'], tree_commit=run['tree']['commit'], run_directory=run['run_directory'])
        if name == 'addendum':value['addendum_to'] = dict(value['addendum_to'], sha256=shas['amendment'])
        if name == 'both':value['addendum_to'] = [dict(path='a', sha256=shas['amendment']), dict(path='b', sha256=shas['addendum'])]
        if name == 'extension':value['addendum_to'] = [dict(path=k, sha256=shas[k]) for k in ('amendment', 'addendum', 'both')]
        raw = json.dumps(value, sort_keys=True).encode();shas[name] = hashlib.sha256(raw).hexdigest();out[name] = (value, shas[name])
    return out


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root, self.run, self.d, self.lad = T.make_run(self.directory.name)
        T.populate(self.root, self.run, self.lad)
        for arm in ('RAW', 'DN2W8'):   # as PromotionTests: screen cells played, most of the extension pre-claimed
            for cell in panel.cells(self.lad, arm)[:1995]:
                (self.root/'claims').mkdir(exist_ok=True); (self.root/'claims'/f"{cell['ordinal']}.json").write_bytes(b'{}')
        self.files = files_for(self.run)
        self.genuine = panel.readout(self.root, arm='DN2W8', limit=500)
        self.screen = self.pin('screen', self.genuine)

    def tearDown(self): self.directory.cleanup()

    def pin(self, name, value):
        path = Path(self.directory.name)/f'{name}.json'
        if not path.exists(): panel.write_new(path, value)
        return path, panel.sha(value)

    def amended(self, readout=None, screen=None):
        (a, a_sha), (h, h_sha), (b, b_sha) = (self.files[k] for k in ('amendment', 'addendum', 'both'))
        return V.verdict(readout or self.genuine, a, self.root, readout_sha256=(screen or self.screen)[1], amendment_sha256=a_sha,
                         addendum=h, addendum_sha256=h_sha, both=b, both_sha256=b_sha)

    def pin_verdict(self, name, value): return self.pin(name, json.loads(json.dumps(value)))

    def test_the_extension_is_pending_2000s_packets_order_and_attempted_counts(self):
        self.assertEqual(self.genuine['decision'], 'CONTINUE_TO_2000')
        official = panel.interleaved(self.root, search_arm='DN2W8', limit=2000, resume=True, screen=self.screen)
        verdict = self.amended(); self.assertEqual((verdict['decision'], verdict['breakage']), ('CONTINUE_TO_2000', []))
        self.assertEqual(E.gate(self.root, screen=self.screen, verdict=self.pin_verdict('verdict', verdict), files=self.files)['decision'], 'CONTINUE_TO_2000')
        ours = E.extension_interleaved(self.root, search_arm='DN2W8', resume=True)
        self.assertEqual(ours, official); self.assertEqual(len(ours[0]), 10)   # 5 per arm, interleaved RAW then search per pair
        self.assertEqual(E.extension_pending(self.root, arm='DN2W8', resume=True), panel.pending(self.root, arm='DN2W8', limit=2000, resume=True, screen=self.screen))
        with self.assertRaisesRegex(ValueError, 'resume explicitly'): E.extension_pending(self.root, arm='DN2W8')

    def test_refuses_every_verdict_but_a_clean_continue(self):
        verdict = self.amended()
        cases = dict(harm=dict(verdict, decision='STOP_HARM', harm=True), breakage_stop=dict(verdict, decision='STOP_BREAKAGE', breakage=['transport_failures']),
                     void=dict(verdict, decision='VOID_FAIRNESS'), flag=dict(verdict, decision='FLAG_ARM_FAIRNESS'), incomplete=dict(verdict, decision='INCOMPLETE'),
                     continue_with_breakage=dict(verdict, breakage=['non_home']))
        for name, value in cases.items():   # genuine verdicts (their recomputation agrees) that do not continue cleanly
            with self.subTest(name), patch.object(E.V, 'verdict', return_value=value), self.assertRaisesRegex(ValueError, 'continuing to 2000 with no breakage'):
                E.gate(self.root, screen=self.screen, verdict=self.pin_verdict(f'v-{name}', value), files=self.files)

    def test_refuses_forged_mismatched_or_foreign_inputs(self):
        verdict = self.amended(); pinned = self.pin_verdict('verdict', verdict)
        with self.assertRaisesRegex(ValueError, 'amended verdict differs'):   # a stored verdict the readout and files do not give
            E.gate(self.root, screen=self.screen, verdict=self.pin_verdict('forged', dict(verdict, conservative=None)), files=self.files)
        with self.assertRaisesRegex(ValueError, 'immutable ladder input differs'):
            E.gate(self.root, screen=self.screen, verdict=(pinned[0], '0'*64), files=self.files)
        tampered = dict(self.genuine, harm=True)   # a readout the run does not reproduce
        with self.assertRaisesRegex(ValueError, 'screen readout differs from its recomputation'):
            E.gate(self.root, screen=self.pin('tampered', tampered), verdict=pinned, files=self.files)
        with self.assertRaisesRegex(ValueError, 'pinned 500-cell screen readout'):
            E.gate(self.root, screen=self.pin('foreign', dict(self.genuine, declaration_sha256='0'*64)), verdict=pinned, files=self.files)
        broken = dict(self.files, extension=(dict(self.files['extension'][0], addendum_to=[]), self.files['extension'][1]))
        with self.assertRaisesRegex(ValueError, 'extension addendum'): E.gate(self.root, screen=self.screen, verdict=pinned, files=broken)
        with self.assertRaisesRegex(ValueError, 'tree commit'): E.gate(self.root, screen=self.screen, verdict=pinned, files=self.files, tree_commit='0'*40)

    def test_another_run_directory_or_unpinned_files_are_refused(self):
        verdict = self.pin_verdict('verdict', self.amended())
        moved = dict(self.files, amendment=(dict(self.files['amendment'][0], run=dict(self.files['amendment'][0]['run'], run_directory='/out/other')), self.files['amendment'][1]))
        with self.assertRaisesRegex(ValueError, "amended run's directory"): E.gate(self.root, screen=self.screen, verdict=verdict, files=moved)
        with self.assertRaisesRegex(ValueError, 'not the committed amendment file'): E.committed(dict(E.PINNED, both='0'*64))

    def test_the_committed_chain_names_each_earlier_file(self):
        real = E.committed()
        self.assertEqual(sorted(x['sha256'] for x in real['extension'][0]['addendum_to']), sorted(real[k][1] for k in ('amendment', 'addendum', 'both')))
        self.assertEqual(sorted(x['sha256'] for x in real['both'][0]['addendum_to']), sorted(real[k][1] for k in ('amendment', 'addendum')))
        self.assertEqual(real['addendum'][0]['addendum_to']['sha256'], real['amendment'][1])


if __name__ == '__main__':
    unittest.main()
