import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageFont
from pathlib import Path

import cv2
import numpy as np
import torch

from fabric_pipeline import (FabricModel, label_path, load_spec, percentile_scores,
                             pool_region, read_boxes, fit_normal, FabricInference, fit_known_distribution, rejection_reasons, save_annotated_image, load_label_font, fuse_backbone, preprocess_image, feature_batches, check_dataset, split_images)


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

    def test_unknown_distribution_and_ambiguity(self):
        model = FabricModel(2).eval()
        vectors = torch.cat([torch.zeros(2, 48), torch.ones(2, 48)])
        fit_known_distribution(model, vectors, torch.tensor([0, 0, 1, 1]))
        self.assertTrue(torch.isfinite(model.known_radius).all())
        self.assertEqual(rejection_reasons(.99, .98, 10, 1, .65, .15),
                         ['outside_known_distribution'])
        self.assertIn('ambiguous_classes', rejection_reasons(.55, .1, .5, 1, .65, .15))
        self.assertEqual(rejection_reasons(.9, .8, .5, 1, .65, .15), [])

    def test_unseen_anomaly_keeps_box_despite_high_softmax(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FabricModel(2).eval()
            model.reference = torch.tensor([0., 1.])
            model.known_radius.fill_(1.)
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.copy_(torch.tensor([20., 0.]))
            checkpoint = root / 'model.pt'
            torch.save({'state_dict': model.state_dict(), 'names': ['A', 'B'],
                        'size': 64, 'quantile': .995, 'min_area': 1,
                        'class_threshold': .65, 'margin': .15}, checkpoint)
            image = root / 'image.png'
            cv2.imwrite(str(image), np.zeros((64, 64, 3), dtype=np.uint8))
            engine = FabricInference(checkpoint, 'cpu')
            # Controlled unseen feature far from known classes and normal distribution.
            with patch.object(engine.model, 'encode', return_value=torch.full((1, 48, 4, 4), 10.)):
                result = engine.predict(image)
            self.assertTrue(result['is_defect'])
            self.assertEqual(len(result['regions']), 1)
            region = result['regions'][0]
            self.assertEqual(region['bbox_xyxy'], [0, 0, 64, 64])
            self.assertEqual(region['class_name'], '未知类型')
            self.assertGreater(region['class_confidence'], .99)
            self.assertIn('outside_known_distribution', region['unknown_reasons'])

    def test_annotation_preserves_resolution_and_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'original.png'
            cv2.imwrite(str(source), np.full((240, 640, 3), 255, dtype=np.uint8))
            original_bytes = source.read_bytes()
            font = ImageFont.truetype('DejaVuSans.ttf', 18)
            result = {'image': str(source), 'regions': [
                {'bbox_xyxy': [20, 80, 150, 170], 'class_name': 'A',
                 'class_confidence': .9, 'normal_deviation_confidence': .999},
                {'bbox_xyxy': [300, 100, 500, 200], 'class_name': '未知类型',
                 'class_confidence': .99, 'normal_deviation_confidence': .998}]}
            destination = Path(tmp) / 'vis' / 'original.png'
            save_annotated_image(result, destination, font)
            with Image.open(destination) as image:
                self.assertEqual(image.size, (640, 240))
                self.assertEqual(image.getpixel((20, 120)), (255, 40, 40))
                self.assertEqual(image.getpixel((300, 150)), (255, 140, 0))
            self.assertEqual(source.read_bytes(), original_bytes)
            with self.assertRaises(ValueError):
                save_annotated_image(result, source, font)
            result['regions'] = []
            save_annotated_image(result, destination, font)
            with Image.open(destination) as image:
                self.assertEqual(image.getpixel((20, 120)), (255, 255, 255))
            with self.assertRaises(FileNotFoundError):
                load_label_font(str(Path(tmp) / 'missing.ttf'))

    def test_fusion_equivalence_and_batch_features(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = FabricModel(2).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            before = model.encode(x)
            fuse_backbone(model.features)
            after = model.encode(x)
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-4)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in model.features.modules()))
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                path = Path(tmp) / f'{i}.png'
                image = np.full((512, 512, 3), 50+i*30, dtype=np.uint8)
                cv2.imwrite(str(path), image)
                paths.append(path)
            batches = list(feature_batches(model, paths, 64, 'cpu', 2))
            self.assertEqual([len(p) for p, _ in batches], [2, 1])
            with torch.no_grad():
                single = model.encode(preprocess_image(image, 64).unsqueeze(0))
            torch.testing.assert_close(single, batches[-1][1])

    def test_mobilenet_448_feature_grid(self):
        torch.set_num_threads(1)
        model = FabricModel(19).eval()
        with torch.no_grad():
            feature = model.encode(torch.zeros(1, 3, 448, 448))
        self.assertEqual(feature.shape, (1, 48, 28, 28))
        self.assertEqual(model.distance(feature).shape, (1, 28, 28))

    def test_check_uses_filtered_lists_and_rejects_empty_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = {'names': ['A', 'B']}
            for split in ('normaltrain', 'normalval', 'train', 'val'):
                folder = root / split / 'images'
                folder.mkdir(parents=True)
                spec[split] = folder
                for name, content in [('shared', b'same'), ('unique', split.encode())]:
                    (folder / (name + '.png')).write_bytes(content)
                    labels = folder.parent / 'labels'
                    labels.mkdir(exist_ok=True)
                    (labels / (name + '.txt')).write_text('0 .25 .25 .2 .2\n1 .75 .75 .2 .2\n')
            report_path = root / 'duplicates.json'
            report = check_dataset(spec, report_path)
            self.assertEqual(report['val']['images'], 1)
            self.assertEqual(len(split_images(spec, 'normalval')), 1)
            self.assertEqual(len(split_images(spec, 'train')), 2)
            with self.assertRaises(ValueError):
                check_dataset(spec, report_path, strict_duplicates=True)
            (spec['normalval'] / 'unique.png').unlink()
            with self.assertRaisesRegex(ValueError, 'normalval'):
                check_dataset(spec, report_path)

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
