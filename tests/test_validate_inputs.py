"""Exercise the public APIs of validate_inputs.py without GPU or network."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import validate_inputs as inputs


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def config(self, tasks, dry_run=False):
        path = self.root / 'config.json'
        path.write_text(json.dumps(tasks))
        return inputs.validate_config(str(path), dry_run=dry_run)

    def test_t2v_alignment_and_multiple_tasks(self):
        report = self.config([{'width': 961, 'height': 543, 'length': 121}, {}])
        self.assertEqual(report['status'], 'all_passed')
        first, second = report['tasks']
        self.assertEqual(first['aligned_resolution'], '960x544')
        self.assertEqual(first['aligned_frames'], 124)
        self.assertEqual(second['mode'], 't2v')
        self.assertEqual(inputs.align_dimensions_32(961, 543), (960, 544))
        self.assertEqual(inputs.align_frame_count(125), 141)

    def test_missing_config_and_invalid_mode(self):
        with self.assertRaises(FileNotFoundError):
            inputs.validate_config(str(self.root / 'missing.json'))
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            self.config({'mode': 'unsupported'})

    def test_i2v_requires_frame_even_in_dry_run(self):
        with self.assertRaises(ValueError):
            self.config({'mode': 'i2v'}, dry_run=True)
        with self.assertRaises(FileNotFoundError):
            self.config({'mode': 'i2v', 'first_frame': str(self.root / 'missing.png')})
        report = self.config({'mode': 'i2v', 'start_image': 'example.png'}, dry_run=True)
        self.assertEqual(report['tasks'][0]['first_frame'], 'example.png')

    def test_real_first_and_last_frame_checks(self):
        a, b = self.root / 'a.png', self.root / 'b.png'
        Image.new('RGB', (32, 32), 'black').save(a)
        Image.new('RGB', (32, 32), 'white').save(b)
        self.assertEqual(self.config({'mode': 'i2v', 'first_frame': str(a)})['status'], 'all_passed')
        pair = self.config({'mode': 'flf2v', 'start_image': str(a), 'end_image': str(b)})['tasks'][0]['pair_report']
        self.assertTrue(pair['recommend_tone_compensate'])
        Image.new('RGB', (48, 32)).save(b)
        with self.assertRaises(ValueError):
            inputs.validate_flf2v_pair(str(a), str(b))
        with self.assertRaises(ValueError):
            self.config({'mode': 'flf2v', 'first_frame': str(a)}, dry_run=True)
        Image.new('RGB', (33, 32)).save(a)
        Image.new('RGB', (33, 32)).save(b)
        with self.assertRaises(ValueError):
            inputs.validate_flf2v_pair(str(a), str(b))

    def test_ref2va_quota_and_duration_rejections(self):
        invalid = [dict(images=[], videos=[], audios=['a.wav']),
                   dict(images=['i'] * 10, videos=[], audios=[]),
                   dict(images=['i'], videos=[{'duration': 3}] * 4, audios=[]),
                   dict(images=['i'], videos=[], audios=[{'duration': 3}] * 4),
                   dict(images=['i'] * 9, videos=[{'duration': 3}] * 3, audios=[{'duration': 3}]),
                   dict(images=['i'], videos=[{'duration': 1}], audios=[]),
                   dict(images=['i'], videos=[{'duration': 16}], audios=[]),
                   dict(images=['i'], videos=[{'duration': 8}] * 2, audios=[]),
                   dict(images=['i'], videos=[], audios=[{'duration': 1}]),
                   dict(images=['i'], videos=[], audios=[{'duration': 16}]),
                   dict(images=['i'], videos=[], audios=[{'duration': 8}] * 2)]
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                inputs.validate_ref2va_quotas(**case)

    def test_ref2va_valid_boundary_and_audio_binding_order(self):
        report = self.config({'mode': 'ref2va', 'ref_images': ['i'] * 9,
                              'ref_videos': [{'duration': 7.5}] * 2,
                              'ref_audios': [{'duration': 15}],
                              'prompt': '@图片1 @视频1 @音频1 @音频3'})
        quota = report['tasks'][0]['quota_report']
        self.assertEqual(quota['counts']['total_files'], 12)
        self.assertEqual(quota['durations']['videos_total_sec'], 15)
        self.assertEqual(quota['bindings_map']['audio_tracks'][-1]['index'], 3)
        self.assertEqual(quota['bindings_map']['audio_tracks'][0]['ref_type'], 'video_audio')
        self.assertEqual(quota['detected_prompt_tags'], ['<Picture 1>', '<Video 1>', '<Audio 1>', '<Audio 3>'])
