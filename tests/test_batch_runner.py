"""Offline regression coverage for prompt, schema and media onboarding contracts."""
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import batch_runner as runner


class PromptTests(unittest.TestCase):
    def test_plain_and_reference_tags(self):
        self.assertEqual(runner.format_h3_prompt({'prompt': '@图片1 follows @视频2 with @音频3'}),
                         '<Picture 1> follows <Video 2> with <Audio 3>')
        self.assertEqual(runner.format_h3_prompt('plain'), 'plain')
        self.assertEqual(runner.format_h3_prompt({}), '')

    def test_structured_shots_and_audio(self):
        task = {'shots': [{'time': '0', 'description': '@图片1 turns'},
                          {'time': '00:03.000', 'prompt': 'Pull back'}, 'Hold'],
                'overall_soundscape': 'Ticks', 'non_diegetic_music': 'Soft strings'}
        self.assertEqual(runner.format_h3_prompt(task),
                         'integrated_multimodal_description:\n[Shot 1] <Picture 1> turns\n'
                         '[Shot 2] At 00:03.000, Pull back\n[Shot 3] At 00:10.000, Hold\n\n'
                         'overall_soundscape: Ticks\n\nnon_diegetic_music: Soft strings')

    def test_prompt_with_only_soundscape(self):
        self.assertEqual(runner.format_h3_prompt({'prompt': 'Street', 'overall_soundscape': 'Rain'}),
                         'integrated_multimodal_description:\nStreet\n\noverall_soundscape: Rain')

    def test_frame_grid_boundaries(self):
        for value, expected in [(-1, 5), (0, 5), (5, 5), (6, 22), (121, 124), (124, 124), (125, 141)]:
            with self.subTest(value=value):
                self.assertEqual(runner.align_frame_count(value), expected)
        for value in range(5, 400):
            aligned = runner.align_frame_count(value)
            self.assertGreaterEqual(aligned, value)
            self.assertLess(aligned - value, 17)
            self.assertEqual((aligned - 5) % 17, 0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.wf = {'6': {'class_type': 'TestNode', 'inputs': {
            'prompt': 'Watch', 'width': 960, 'legacy_option': True,
            'ref_images.ref_image_0': ['20', 0], 'custom.value': 3}}}
        self.info = {'TestNode': {'input': {'required': {'prompt': ['STRING']},
                                          'optional': {'custom': ['ANY']}}}}

    def test_removes_only_nonessential_and_does_not_mutate(self):
        before = copy.deepcopy(self.wf)
        adapted = runner.validate_and_adapt_workflow(self.wf, self.info)
        self.assertEqual(self.wf, before)
        self.assertNotIn('legacy_option', adapted['6']['inputs'])
        for key in ('prompt', 'width', 'ref_images.ref_image_0', 'custom.value'):
            self.assertIn(key, adapted['6']['inputs'])

    def test_unknown_node_and_missing_required_raise(self):
        with self.assertRaisesRegex(ValueError, 'TestNode'):
            runner.validate_and_adapt_workflow(self.wf, {'Other': {}})
        self.info['TestNode']['input']['required']['clip'] = ['CLIP']
        with self.assertRaisesRegex(ValueError, 'clip'):
            runner.validate_and_adapt_workflow(self.wf, self.info)

    def test_unavailable_schema_leaves_workflow_intact(self):
        self.assertEqual(runner.validate_and_adapt_workflow(self.wf, None), self.wf)


class ShowcaseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(runner, 'PROJECT_ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_missing_showcase_is_generated_and_staged_once(self):
        rel = Path('outputs/showcase/multimodal/input.png')
        def generate(*args, **kwargs):
            (self.root / rel).parent.mkdir(parents=True)
            (self.root / rel).write_bytes(b'example input')
        with patch.object(runner.subprocess, 'run', side_effect=generate) as call:
            for _ in range(2):
                self.assertEqual(runner.stage_media_file(str(rel), self.root / 'input'), 'input.png')
            call.assert_called_once_with(
                [sys.executable, str(self.root / 'scripts/generate_showcase_assets.py')],
                cwd=self.root, check=True)
        self.assertEqual((self.root / 'input/input.png').read_bytes(), b'example input')

    def test_user_files_and_traversal_do_not_trigger_generation(self):
        with patch.object(runner.subprocess, 'run') as call:
            for name in ('custom.png', 'outputs/showcase/../custom.png'):
                with self.assertRaises(FileNotFoundError):
                    runner.stage_media_file(name, self.root / 'input')
            call.assert_not_called()

    def test_generator_failure_is_actionable(self):
        with patch.object(runner.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'generator')):
            with self.assertRaisesRegex(RuntimeError, 'Pillow and ffmpeg'):
                runner.stage_media_file('outputs/showcase/input.png', self.root / 'input')

    def test_generator_missing_requested_file_still_raises(self):
        with patch.object(runner.subprocess, 'run'):
            with self.assertRaises(FileNotFoundError):
                runner.stage_media_file('outputs/showcase/unknown.png', self.root / 'input')
