"""Authored protocol lines and pinned pure-filter parity; no games or models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unittest

from pokezero.local_showdown import default_showdown_root
from pokezero.mcts_eval.search_over_raw_protocol_channels import (
    FILTER_SHA256, MAXIMUM_BYTES, channel_lines, channel_zero_lines,
)


class ProtocolChannelTests(unittest.TestCase):
    def test_spectator_and_each_actor_keep_only_their_visible_split_payloads(self):
        lines = ["|start", "|split|p1", "|debug|P1_PRIVATE_CANARY", "|message|p1-public",
            "|split|p2", "|debug|P2_PRIVATE_CANARY", "|message|p2-public", "|turn|2"]
        original = deepcopy(lines)
        self.assertEqual(channel_zero_lines(lines),
            ["|start", "|message|p1-public", "|message|p2-public", "|turn|2"])
        self.assertEqual(channel_lines(lines, channel=1),
            ["|start", "|debug|P1_PRIVATE_CANARY", "|message|p2-public", "|turn|2"])
        self.assertEqual(channel_lines(lines, channel=2),
            ["|start", "|message|p1-public", "|debug|P2_PRIVATE_CANARY", "|turn|2"])
        self.assertEqual(lines, original)

    def test_ordinary_order_duplicates_and_empty_message_rules_are_preserved(self):
        lines = ["", "|turn|1", "|turn|1", "|split|p1", "", "|message|shared", ""]
        self.assertEqual(channel_zero_lines(lines), ["|turn|1", "|turn|1", "|message|shared"])
        self.assertEqual(channel_lines(lines, channel=1), ["|turn|1", "|turn|1"])
        self.assertEqual(channel_lines([], channel=2), [])

    def test_malformed_truncated_and_nested_split_groups_refuse(self):
        families = [["|split|p1"], ["|split|p2", "secret"],
            ["|split|p3", "secret", "shared"], ["|split|garbage", "secret", "shared"],
            ["|split|p1", "|split|p2", "shared"], ["|split|p1", "secret", "|split|p2"]]
        for lines in families:
            for channel in (0, 1, 2):
                with self.subTest(lines=lines, channel=channel), self.assertRaises(ValueError):
                    channel_lines(lines, channel=channel)

    def test_requests_refuse_even_when_the_selected_channel_would_discard_them(self):
        for lines in (["|request|{}"], ["|split|p1", "|request|private", "shared"],
                      ["|split|p2", "private", "|request|public"]):
            for channel in (0, 1, 2):
                with self.subTest(lines=lines, channel=channel), self.assertRaises(ValueError):
                    channel_lines(lines, channel=channel)

    def test_exact_line_and_channel_types_refuse_numeric_aliases_and_embedded_lines(self):
        for channel in (True, False, -1, 3, 0.0, "1", None):
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                channel_lines([], channel=channel)
        for lines in (None, ("|start",), [1], [False], ["|turn|1\n|request|{}"],
                      ["|message|\rhidden"], ["|message|\x00hidden"],
                      ["|message|\u2028hidden"], ["|message|\u2029hidden"]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                channel_zero_lines(lines)

    def test_retained_size_counts_utf8_delimiters_and_discarded_private_lines(self):
        exact = ["x" * (MAXIMUM_BYTES - 1)]
        self.assertEqual(channel_zero_lines(exact), exact)
        with self.assertRaises(ValueError):
            channel_zero_lines([exact[0] + "x"])
        with self.assertRaises(ValueError):
            channel_zero_lines(["|split|p1", "é" * (MAXIMUM_BYTES // 2), ""])
        with self.assertRaises(ValueError):
            channel_lines(["|split|p2", "x" * MAXIMUM_BYTES, "public"], channel=1)

    def test_all_three_channels_match_pinned_upstream_on_supported_authored_families(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node is required for the pinned pure-filter comparison")
        path = default_showdown_root() / "dist" / "sim" / "battle.js"
        text = path.read_text()
        start = text.index("const splitRegex = ")
        stop = text.index("\nclass Battle {", start)
        function = text[start:stop]
        self.assertEqual(hashlib.sha256(function.encode()).hexdigest(), FILTER_SHA256)
        families = [[], ["", "|start", ""], ["|split|p1", "SECRET", "|message|visible"],
            ["|split|p2", "SECRET", ""], ["|split|p1", "", "shared"], ["|turn|1", "|turn|1"],
            ["|player|p1|A", "|split|p1", "|-damage|p1a: A|117/253", "|-damage|p1a: A|47/100",
             "|split|p2", "|-damage|p2a: B|19/212", "|-damage|p2a: B|9/100", "|turn|2"]]
        program = function + "\nconst fs=require('fs');const families=JSON.parse(fs.readFileSync(0,'utf8'));" \
            + "process.stdout.write(JSON.stringify(families.map(lines=>extractChannelMessages(lines.join('\\n'),[0,1,2]))));"
        result = subprocess.run([node, "-e", program], input=json.dumps(families),
            text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = json.loads(result.stdout)
        for channel in (0, 1, 2):
            self.assertEqual([channel_lines(lines, channel=channel) for lines in families],
                             [row[str(channel)] for row in expected])
