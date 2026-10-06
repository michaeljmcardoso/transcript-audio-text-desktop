import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

from main import MainWindow


class DesktopUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()

    def tearDown(self):
        self.window.close()

    def test_speaker_name_edits_refresh_all_transcript_views(self):
        self.window.result = {
            "language": "pt",
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "full_text": "[SPEAKER_00] Bom dia.\n\n[SPEAKER_01] Olá.",
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:01",
                    "text": "Bom dia.",
                },
                {
                    "speaker": "SPEAKER_01",
                    "start_hms": "00:00:01",
                    "end_hms": "00:00:02",
                    "text": "Olá.",
                },
            ],
        }
        self.window._build_speaker_editor()
        self.window._refresh_outputs()
        self.window.name_edits["SPEAKER_00"].setText("Entrevistadora")

        self.assertIn("Entrevistadora: Bom dia.", self.window.speaker_text.toPlainText())
        self.assertIn("Entrevistadora: Bom dia.", self.window.plain_text.toPlainText())
        self.assertIn(
            "[00:00:00 → 00:00:01] Entrevistadora: Bom dia.",
            self.window.timestamp_text.toPlainText(),
        )

    def test_hugging_face_token_is_not_requested_in_the_ui(self):
        self.assertEqual(
            self.window.gemini_key_edit.echoMode(),
            QLineEdit.EchoMode.Password,
        )
        self.assertFalse(hasattr(self.window, "hf_token_edit"))
        self.assertFalse(hasattr(self.window, "hf_token_help"))
        self.assertFalse(self.window.suggest_button.isEnabled())
        self.assertFalse(self.window.gemini_key_edit.text())

    @patch("main.QMessageBox.warning")
    def test_transcription_requires_environment_token(self, warning):
        self.window.audio_path = Path("sample.wav")
        self.window.transcribe_button.setEnabled(True)

        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            self.window._start_transcription()

        warning.assert_called_once()
        self.assertIn("HF_TOKEN", warning.call_args.args[2])
        self.assertIsNone(self.window.transcription_worker)

    @patch("main.TranscriptionWorker")
    def test_transcription_worker_receives_environment_token(self, worker_class):
        worker = worker_class.return_value
        worker.progress = MagicMock()
        worker.succeeded = MagicMock()
        worker.failed = MagicMock()
        worker.cancelled = MagicMock()
        worker.finished = MagicMock()
        worker.isRunning.return_value = False
        self.window.audio_path = Path("sample.wav")
        self.window.transcribe_button.setEnabled(True)

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            self.window._start_transcription()

        worker_class.assert_called_once_with("sample.wav", "hf_test_token", None)
        worker.start.assert_called_once()

    @patch("main.HF_TOKEN_FOR_TEST", "hf_embedded_test_token")
    @patch("main.TranscriptionWorker")
    def test_local_test_token_overrides_environment_token(self, worker_class):
        worker = worker_class.return_value
        worker.progress = MagicMock()
        worker.succeeded = MagicMock()
        worker.failed = MagicMock()
        worker.cancelled = MagicMock()
        worker.finished = MagicMock()
        worker.isRunning.return_value = False
        self.window.audio_path = Path("sample.wav")
        self.window.transcribe_button.setEnabled(True)

        with patch.dict(os.environ, {"HF_TOKEN": "hf_environment_token"}):
            self.window._start_transcription()

        worker_class.assert_called_once_with(
            "sample.wav",
            "hf_embedded_test_token",
            None,
        )
        worker.start.assert_called_once()

    @patch("main.TranscriptionWorker")
    def test_cancel_button_requests_nonblocking_worker_cancellation(self, worker_class):
        worker = worker_class.return_value
        worker.isRunning.return_value = True
        self.window.transcription_worker = worker

        self.window._request_transcription_cancel()

        worker.request_cancel.assert_called_once()
        self.assertFalse(self.window.cancel_button.isEnabled())
        self.assertIn("Cancelamento solicitado", self.window.status_label.text())
        self.window.transcription_worker = None

    @patch("main.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes)
    def test_closing_during_transcription_requests_cancel_without_waiting(self, _question):
        worker = Mock()
        worker.isRunning.return_value = True
        self.window.transcription_worker = worker
        event = Mock()

        self.window.closeEvent(event)

        worker.request_cancel.assert_called_once()
        worker.wait.assert_not_called()
        event.ignore.assert_called_once()
        self.assertTrue(self.window._close_when_workers_idle)
        self.window._close_when_workers_idle = False
        self.window.transcription_worker = None

    def test_export_buttons_offer_speaker_docx_and_timestamps_txt(self):
        self.assertEqual(self.window.docx_button.text(), "Baixar por Falante (DOCX)")
        self.assertEqual(
            self.window.timestamps_button.text(),
            "Baixar Marcas de Tempo (TXT)",
        )
        self.assertFalse(hasattr(self.window, "txt_button"))

    def test_message_dialog_colors_remain_readable_in_system_dark_mode(self):
        stylesheet = self.window.styleSheet()

        self.assertIn("QMessageBox {", stylesheet)
        self.assertIn("background-color: #FFFFFF;", stylesheet)
        self.assertIn("QMessageBox QLabel {", stylesheet)
        self.assertIn("color: #202124;", stylesheet)

    def test_dropdown_and_context_menu_colors_remain_readable_in_dark_mode(self):
        stylesheet = self.window.styleSheet()

        self.assertIn("QComboBox QAbstractItemView {", stylesheet)
        self.assertIn("QMenu {", stylesheet)
        self.assertIn("QMenu::item:selected {", stylesheet)
        self.assertIn("selection-color: #FFFFFF;", stylesheet)

    def test_local_name_suggestions_are_applied_after_transcription(self):
        result = {
            "language": "pt",
            "speakers": ["SPEAKER_00", "SPEAKER_01"],
            "full_text": (
                "[SPEAKER_00] Eu sou a Maria, prazer.\n\n"
                "[SPEAKER_01] Bom dia."
            ),
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0,
                    "end": 1,
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:01",
                    "text": "Eu sou a Maria, prazer.",
                },
                {
                    "speaker": "SPEAKER_01",
                    "start": 1,
                    "end": 2,
                    "start_hms": "00:00:01",
                    "end_hms": "00:00:02",
                    "text": "Bom dia.",
                },
            ],
        }

        self.window._transcription_succeeded(result)

        self.assertEqual(self.window.name_edits["SPEAKER_00"].text(), "Maria")
        self.assertEqual(self.window.name_edits["SPEAKER_01"].text(), "")
        self.assertIn("sem enviar dados", self.window.suggestion_status.text())
        self.assertTrue(self.window.status_label.font().bold())

    def test_automatic_suggestions_do_not_overwrite_manual_name_edits(self):
        self.window.result = {
            "language": "pt",
            "speakers": ["SPEAKER_00"],
            "full_text": "[SPEAKER_00] Eu sou a Maria.",
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0,
                    "end": 1,
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:01",
                    "text": "Eu sou a Maria.",
                },
            ],
        }
        self.window._build_speaker_editor()
        self.window._apply_suggestions({"SPEAKER_00": "Maria"})
        self.window.name_edits["SPEAKER_00"].setText("Mariana")

        self.window._apply_suggestions({"SPEAKER_00": "Maria"})

        self.assertEqual(self.window.name_edits["SPEAKER_00"].text(), "Mariana")


if __name__ == "__main__":
    unittest.main()
