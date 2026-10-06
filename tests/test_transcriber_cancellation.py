import unittest
from unittest.mock import patch

import transcriber


class TranscriberCancellationTests(unittest.TestCase):
    def test_cancellation_before_processing_prevents_audio_setup(self):
        with patch.object(transcriber, "ensure_ffmpeg_available") as ensure_ffmpeg:
            with self.assertRaises(transcriber.TranscriptionCancelled):
                transcriber.transcribe_and_diarize(
                    audio_path="sample.wav",
                    hf_token="hf_test_token",
                    cancellation_callback=lambda: True,
                )

        ensure_ffmpeg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
