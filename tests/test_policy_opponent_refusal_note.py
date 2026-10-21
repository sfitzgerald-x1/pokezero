"""Authored refusal diagnostics only; no historical decision is replayed."""
import json
import traceback
import unittest

from pokezero.engine_search import _note_policy_opponent_refusal


class PolicyOpponentRefusalNoteTests(unittest.TestCase):
    def diagnostic(self):
        return {"schema":"policy-opponent-refusal-v1","kind":"legal_surface_mismatch",
            "diagnostic_only_not_policy_input":True,"seat":"p2","root":False,
            "native_action_indices":[0],"private_action_indices":[0,4,5],
            "active_index":1,"node_depth":2,"force_switch":False,"waiting":False,
            "recharging":False,"own_volatile_statuses":"{}","own_last_used_move":"MOVE:0",
            "sampled_own_side":"PRIVATE_OWN_PAYLOAD","opposing_state":"FORBIDDEN_OPPONENT_PAYLOAD",
            "request_order":["OMITTED_IDENTITY"],"public_branch_lines":["OMITTED_BRANCH"]}

    def error(self,diagnostic=None):
        error = ValueError("native and private-knowledge legal surfaces differ")
        error.policy_opponent_diagnostic = json.dumps(self.diagnostic() if diagnostic is None else diagnostic)
        return error

    def test_safe_facts_are_visible_in_chained_traceback_without_state_payloads(self):
        original = self.error()
        _note_policy_opponent_refusal(original)
        try:
            raise RuntimeError("world refused") from original
        except RuntimeError as outer:
            output = "".join(traceback.format_exception(outer))
            self.assertIs(outer.__cause__,original)
        self.assertIn('"native_action_indices": [0]',output)
        self.assertIn('"private_action_indices": [0, 4, 5]',output)
        self.assertIn('"node_depth": 2',output)
        for secret in ("PRIVATE_OWN_PAYLOAD","FORBIDDEN_OPPONENT_PAYLOAD","OMITTED_IDENTITY","OMITTED_BRANCH"):
            self.assertNotIn(secret,output)
        self.assertEqual(str(original),"native and private-knowledge legal surfaces differ")

    def test_malformed_unrecognized_and_oversized_diagnostics_do_not_replace_failure(self):
        for raw in (None,{},"not json","[]",'"text"',"x"*131073):
            error = ValueError("original")
            error.policy_opponent_diagnostic = raw
            _note_policy_opponent_refusal(error)
            self.assertEqual(str(error),"original")
            self.assertFalse(hasattr(error,"__notes__"))
        for changes in ({"schema":"unknown"},{"kind":"other"},{"diagnostic_only_not_policy_input":False}):
            d = self.diagnostic() | changes
            e = self.error(d)
            _note_policy_opponent_refusal(e)
            self.assertFalse(hasattr(e,"__notes__"))

    def test_fields_are_type_and_size_bounded_and_control_characters_escaped(self):
        d = self.diagnostic() | {"root":{},"active_index":True,"node_depth":-1,
            "native_action_indices":[True],"private_action_indices":list(range(10)),
            "own_volatile_statuses":"x"*2049,"own_last_used_move":"MOVE\nnext\x1bline"}
        e = self.error(d)
        _note_policy_opponent_refusal(e)
        note, = e.__notes__
        self.assertNotIn("\n",note)
        self.assertNotIn("\x1b",note)
        parsed = json.loads(note.split(": ",1)[1])
        for key in ("root","active_index","node_depth","native_action_indices","private_action_indices","own_volatile_statuses"):
            self.assertNotIn(key,parsed)

    def test_missing_or_broken_optional_attribute_never_masks_failure(self):
        class Broken(ValueError):
            @property
            def policy_opponent_diagnostic(self):
                raise RuntimeError("instrumentation unavailable")
        for e in (ValueError("original"),Broken("original")):
            _note_policy_opponent_refusal(e)
            self.assertEqual(str(e),"original")
            self.assertFalse(hasattr(e,"__notes__"))


if __name__ == "__main__":
    unittest.main()
