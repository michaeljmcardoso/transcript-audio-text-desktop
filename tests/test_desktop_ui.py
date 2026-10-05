import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit

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

    def test_credentials_are_password_fields_and_suggestion_is_opt_in(self):
        self.assertEqual(
            self.window.hf_token_edit.echoMode(),
            QLineEdit.EchoMode.Password,
        )
        self.assertEqual(
            self.window.gemini_key_edit.echoMode(),
            QLineEdit.EchoMode.Password,
        )
        self.assertFalse(self.window.suggest_button.isEnabled())
        self.assertFalse(self.window.gemini_key_edit.text())

    def test_export_buttons_offer_speaker_docx_and_timestamps_txt(self):
        self.assertEqual(self.window.docx_button.text(), "Baixar por Falante (DOCX)")
        self.assertEqual(self.window.timestamps_button.text(), "Baixar Timestamps (TXT)")
        self.assertFalse(hasattr(self.window, "txt_button"))


if __name__ == "__main__":
    unittest.main()
