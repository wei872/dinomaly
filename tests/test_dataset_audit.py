import json
import tempfile
import unittest
from pathlib import Path

from dataset_audit import audit_duplicates, filter_cross_split_duplicates


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

    def test_skip_validation_keep_training_and_unique_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = {}
            for split in ('normaltrain', 'train', 'normalval', 'val'):
                folder = root / split
                folder.mkdir()
                shared, unique = folder / 'shared.png', folder / 'unique.png'
                shared.write_bytes(b'same')
                unique.write_bytes(split.encode())
                splits[split] = [shared, unique]
            report = audit_duplicates(splits, root / 'report.json')
            filtered, skipped = filter_cross_split_duplicates(splits, report)
            self.assertEqual(len(skipped), 2)
            self.assertEqual(len(filtered['normaltrain']), 2)
            self.assertEqual(len(filtered['train']), 2)
            self.assertEqual(len(filtered['normalval']), 1)
            self.assertEqual(len(filtered['val']), 1)
            self.assertTrue(all(p.exists() for paths in splits.values() for p in paths))
            self.assertTrue(all(item['split'] in ('normalval', 'val') for item in skipped))

    def test_validation_only_overlap_prefers_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = root / 'a.png', root / 'b.png'
            a.write_bytes(b'same'); b.write_bytes(b'same')
            splits = {'normalval': [a], 'val': [b]}
            report = audit_duplicates(splits, root / 'report.json')
            filtered, skipped = filter_cross_split_duplicates(splits, report)
            self.assertEqual(filtered['normalval'], [a])
            self.assertEqual(filtered['val'], [])
            self.assertEqual(len(skipped), 1)

    def test_within_split_only_not_cross_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b, c = [root / name for name in ['a', 'b', 'c']]
            a.write_bytes(b'same'); b.write_bytes(b'same'); c.write_bytes(b'other')
            report = audit_duplicates({'train': [a, b], 'val': [c]}, root / 'report.json')
            self.assertEqual(report['duplicate_group_count'], 0)


if __name__ == '__main__':
    unittest.main()
