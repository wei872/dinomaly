import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from fabric_pipeline import (FabricModel, label_path, load_spec, percentile_scores,
                             pool_region, read_boxes, fit_normal, FabricInference)


class FabricTests(unittest.TestCase):
    def test_yolo_multiclass_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.txt'
            p.write_text('0 .5 .5 .2 .4\n18 .9 .9 .2 .2\n')
            boxes = read_boxes(p, 19)
            self.assertEqual([b[0] for b in boxes], [0, 18])
            self.assertAlmostEqual(boxes[0][1], .4)
            p.write_text('')
            self.assertEqual(read_boxes(p, 19), [])
            p.write_text('19 .5 .5 .2 .2')
            with self.assertRaises(ValueError):
                read_boxes(p, 19)

    def test_label_nested_path(self):
        self.assertEqual(label_path(Path('/data/images/train/sub/a.jpg')),
                         Path('/data/labels/train/sub/a.txt'))

    def test_percentile_absolute_reference(self):
        scores = percentile_scores(np.array([0, 2, 9]), np.array([1, 2, 3, 4]))
        np.testing.assert_allclose(scores, [0, .5, 1])

    def test_windows_yaml_relocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sub in ['images/train', 'images/val', 'normal/images/train', 'normal/images/val']:
                (root / sub).mkdir(parents=True)
            spec = load_spec('dataset/fabric19.yaml', tmp)
            self.assertEqual(spec['normaltrain'], root / 'normal/images/train')
            self.assertEqual(len(spec['names']), 19)

    def test_normal_fit_checkpoint_and_predict(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, value in [('normaltrain', 120), ('normalval', 130)]:
                (root / split).mkdir()
                cv2.imwrite(str(root / split / 'sample.png'), np.full((64, 64, 3), value, dtype=np.uint8))
            model = FabricModel(19).eval()  # Random weights: plumbing test, not accuracy.
            old_mean = model.mean.clone()
            fit_normal(model, {'normaltrain': root / 'normaltrain',
                               'normalval': root / 'normalval'}, 64, 'cpu')
            self.assertFalse(torch.equal(old_mean, model.mean))
            self.assertTrue(torch.all(model.variance > 0))
            self.assertTrue(torch.isfinite(model.reference).all())
            checkpoint = root / 'model.pt'
            torch.save({'state_dict': model.state_dict(), 'names': [str(i) for i in range(19)],
                        'size': 64, 'quantile': .995, 'min_area': 1,
                        'class_threshold': 1., 'margin': .15}, checkpoint)
            engine = FabricInference(checkpoint, 'cpu')
            np.testing.assert_allclose(engine.model.mean.numpy(), model.mean.numpy())
            result = engine.predict(root / 'normalval' / 'sample.png')
            self.assertGreaterEqual(result['normal_deviation_confidence'], 0)
            self.assertLessEqual(result['normal_deviation_confidence'], 1)
            for region in result['regions']:
                self.assertEqual(region['class_name'], '未知类型')

    def test_backbone_and_region(self):
        model = FabricModel(19).eval()
        with torch.no_grad():
            f = model.encode(torch.zeros(1, 3, 64, 64))
        self.assertEqual(f.shape, (1, 48, 4, 4))
        self.assertEqual(pool_region(f, (.99, .99, 1., 1.)).shape, (48,))
        self.assertEqual(model.distance(f).shape, (1, 4, 4))
        self.assertTrue(torch.isfinite(model.distance(f)).all())


if __name__ == '__main__':
    unittest.main()
