"""Small actual NPZ banks: content overlap, not seed/index equality."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
from gpu_engine.team_pool import pack_stream
from pokezero import b4_canary_team_validation as v
from pokezero.b4_fidelity_canary import file_sha
from test_gpu_engine_team_pool import fixture


class TeamValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def bank(self, levels, *, seed=100, permute=False, renumber=False):
        rows = copy.deepcopy(fixture())
        rows[0].update(team_count=len(levels), seed_start=seed)
        prototype = rows[1]
        records = [rows[0]]
        for i, level in enumerate(levels):
            row = copy.deepcopy(prototype)
            row.update(index=i, seed=seed+i)
            for slot, member in enumerate(row['members']):
                member['level'] = level
                member['stats'][0] += slot
                if permute:
                    member['moves'].reverse();member['pp'].reverse();member['types'].reverse()
            if permute:row['members'].reverse()
            records.append(row)
        records.append(dict(kind='terminal', team_count=len(levels)))
        # Pack validates primary types, so permute types after packing below.
        if permute:
            for row in records[1:-1]:
                for member in row['members']:member['types'].reverse()
        arrays, metadata = pack_stream(io.StringIO(''.join(json.dumps(r)+'\n' for r in records)), count=len(levels), seed_start=seed)
        if permute:arrays['types'] = arrays['types'][:, :, ::-1]
        if renumber:
            for field, name in v.VOCABULARIES.items():
                arrays[field] = (len(metadata['vocabulary'][name])-1-arrays[field]).astype(arrays[field].dtype)
            for values in metadata['vocabulary'].values():values.reverse()
            arrays['metadata_utf8'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
        path = self.root / (str(self.serial)+'.npz');self.serial += 1
        np.savez(path, **arrays)
        return dict(path=str(path), sha256=file_sha(path))

    def check(self, candidate, refs):
        game = NS(team_indices=(0, 1))
        plan = NS(pool_sha256=candidate['sha256'], iterations=((NS(games=(game,)),),))
        return v.validate_team_banks(plan, candidate['path'], refs)

    def test_disjoint_contents_accept_even_with_same_seed_and_indices(self):
        candidate = self.bank([70, 71])
        refs = {role:[self.bank([80,81])] for role in v.REFERENCE_ROLES}
        self.assertEqual(self.check(candidate, refs)['status'], 'DISJOINT')

    def test_each_reference_role_rejects_overlap_in_both_directions(self):
        a = self.bank([70,71]);b = self.bank([72,70], seed=900, permute=True, renumber=True)
        disjoint = self.bank([80,81])
        for candidate, overlap in ((a,b),(b,a)):
            for role in v.REFERENCE_ROLES:
                with self.subTest(role=role, candidate=candidate['path']):
                    refs = {key:[disjoint] for key in v.REFERENCE_ROLES};refs[role] = [disjoint,overlap]
                    with self.assertRaisesRegex(ValueError, 'team content overlap: '+role):self.check(candidate, refs)

    def test_missing_empty_changed_banks_and_duplicate_candidate_fail_closed(self):
        candidate = self.bank([70,71]);other = self.bank([80,81])
        refs = {role:[other] for role in v.REFERENCE_ROLES}
        for invalid in ({},dict(refs, r1=[]),dict(refs, r2=[dict(other, sha256='0'*64)])):
            with self.assertRaises(ValueError):self.check(candidate, invalid)
        with self.assertRaisesRegex(ValueError, 'digest'):self.check(dict(candidate,sha256='0'*64), refs)
        with self.assertRaisesRegex(ValueError, 'content reuse'):self.check(self.bank([70,70]), refs)
        with self.assertRaisesRegex(ValueError, 'outside bank'):self.check(self.bank([70]), refs)
        Path(other['path']).unlink()
        with self.assertRaises(FileNotFoundError):self.check(candidate, refs)


if __name__ == '__main__':unittest.main()
