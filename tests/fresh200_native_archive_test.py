import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gpu_engine.fresh200_native_archive import _ClosedStream, _artifact


class ClosedRecordingStreamTest(unittest.TestCase):
    def reader(self, data, *, sealed=None):
        sealed = data if sealed is None else sealed
        path = Path('stdout.bin')
        return _ClosedStream(io.BytesIO(data), {
            'path': str(path), 'bytes': len(sealed), 'sha256': hashlib.sha256(sealed).hexdigest(),
        }, path)

    def test_contiguous_extents_include_empty_command(self):
        reader = self.reader(b'{}\n[]\n')
        self.assertEqual(reader.consume((0, 3)), [{}])
        self.assertEqual(reader.consume((3, 3)), [])
        self.assertEqual(reader.consume((3, 6)), [[]])
        reader.finish()

    def test_offset_gap_is_refused(self):
        reader = self.reader(b'{}\n[]\n')
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            reader.consume((3, 6))

    def test_parseable_mutation_and_unaccounted_tail_are_refused(self):
        for wire in (b'[]\n', b'{}\n[]\n'):
            with self.subTest(wire=wire):
                reader = self.reader(wire, sealed=b'{}\n')
                reader.consume((0, 3))
                with self.assertRaisesRegex(ValueError, 'tail or a changed digest'):
                    reader.finish()


class RecordingMountTest(unittest.TestCase):
    def test_only_the_declared_producer_path_binds_to_durable_bytes(self):
        with TemporaryDirectory() as directory:
            durable = Path(directory)
            producer = Path('/out/recording/command-0')
            name = 'stdin.jsonl'
            data = b'{}\n'
            (durable / name).write_bytes(data)
            reference = dict(path=str(producer / name), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            self.assertEqual(_artifact(durable, producer, name, reference), data)
            with self.assertRaisesRegex(ValueError, 'artifact path differs'):
                _artifact(durable, producer.parent / 'command-1', name, reference)
