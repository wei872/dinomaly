import json
import tempfile
import unittest
from pathlib import Path

from dataset_audit import audit_duplicates


class AuditTests(unittest.TestCase):
    def test_all_cross_split_groups_and_no_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = {}
            for split in ('normaltrain', 'normalval', 'train', 'val'):
                folder = root / split
                folder.mkdir()
                splits[split] = []
                for i in range(2):
                    p = folder / f'{split}_{i}.bmp'
                    p.write_bytes(bytes([i]))
                    splits[split].append(p)
            before = {p: p.read_bytes() for paths in splits.values() for p in paths}
            report_path = root / 'report.json'
            report = audit_duplicates(splits, report_path)
            self.assertEqual(report['duplicate_group_count'], 2)
            self.assertTrue(all(len(g['files']) == 4 for g in report['duplicates']))
            self.assertEqual(json.loads(report_path.read_text()), report)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            with self.assertRaises(ValueError):
                audit_duplicates(splits, splits['train'][0])

    def test_within_split_only_not_cross_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = [root / name for name in ['a', 'b', 'c']]
            a.write_bytes(b'same'); b.write_bytes(b'same'); c.write_bytes(b'other')
            report = audit_duplicates({'train': [a, b], 'val': [c]}, root / 'report.json')
            self.assertEqual(report['duplicate_group_count'], 0)


if __name__ == '__main__':
    unittest.main()
