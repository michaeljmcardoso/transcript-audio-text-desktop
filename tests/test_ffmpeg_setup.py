import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

from ffmpeg_setup import ensure_ffmpeg_available


class FfmpegSetupTests(unittest.TestCase):
    def test_bundled_ffmpeg_is_on_path_and_can_run(self):
        executable = ensure_ffmpeg_available()

        resolved = shutil.which("ffmpeg")
        self.assertIsNotNone(resolved)
        self.assertEqual(os.path.realpath(resolved), executable)
        result = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("ffmpeg version", result.stdout.lower())

    @patch("ffmpeg_setup.imageio_ffmpeg.get_ffmpeg_exe", return_value="/missing/ffmpeg")
    def test_raises_clear_error_if_bundled_executable_is_missing(self, _mock_get_exe):
        with self.assertRaisesRegex(FileNotFoundError, "imageio-ffmpeg"):
            ensure_ffmpeg_available()


if __name__ == "__main__":
    unittest.main()
