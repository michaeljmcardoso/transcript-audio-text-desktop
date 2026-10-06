"""Desktop interface for local WhisperX transcription."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gemini_helper import (
    GeminiRequestError,
    detectar_nomes_falantes,
    sugerir_nomes_locais,
)
from transcript_formatter import format_transcript_by_speaker, transcript_to_docx

AUDIO_EXTENSIONS = "Audio (*.flac *.m4a *.mp3 *.mp4 *.ogg *.opus *.wav *.webm)"
# Teste local apenas: preencher este valor embute o token no executável.
HF_TOKEN_FOR_TEST = ""
APP_STYLESHEET = """
QMainWindow {
    background-color: #F2F6FA;
}
QWidget {
    color: #233547;
    font-family: "Noto Sans";
    font-size: 10pt;
}
QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #D8E2EB;
    border-radius: 9px;
    margin-top: 12px;
    padding: 12px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #31536A;
}
QLineEdit, QComboBox, QTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #CAD7E2;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #168C91;
    selection-color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid #168C91;
}
QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #CAD7E2;
    color: #202124;
    selection-background-color: #168C91;
    selection-color: #FFFFFF;
}
QTextEdit {
    background-color: #F9FBFD;
}
QScrollBar:vertical {
    background-color: #E5EDF3;
    width: 14px;
    margin: 2px;
    border-radius: 7px;
}
QScrollBar::handle:vertical {
    background-color: #8AA6BA;
    min-height: 32px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background-color: #557F99;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
QScrollBar:horizontal {
    background-color: #E5EDF3;
    height: 14px;
    margin: 2px;
    border-radius: 7px;
}
QScrollBar::handle:horizontal {
    background-color: #8AA6BA;
    min-width: 32px;
    border-radius: 6px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #557F99;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}
QPushButton {
    background-color: #E8F0F5;
    border: 1px solid #D0DDE6;
    border-radius: 7px;
    padding: 8px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #DCEBF0;
    border-color: #A9C8D2;
}
QPushButton:pressed {
    background-color: #C9DFE5;
}
QPushButton:disabled {
    background-color: #E9EDF1;
    border-color: #E0E5E9;
    color: #8996A2;
}
QPushButton#primaryAction {
    background-color: #167D83;
    border-color: #167D83;
    color: #FFFFFF;
}
QPushButton#primaryAction:hover {
    background-color: #126B71;
}
QPushButton#exportAction {
    background-color: #315B78;
    border-color: #315B78;
    color: #FFFFFF;
}
QPushButton#exportAction:hover {
    background-color: #274B64;
}
QMessageBox {
    background-color: #FFFFFF;
    color: #202124;
}
QMessageBox QLabel {
    background-color: transparent;
    color: #202124;
}
QMessageBox QPushButton {
    background-color: #E8F0F5;
    border: 1px solid #D0DDE6;
    color: #202124;
}
QMenu {
    background-color: #FFFFFF;
    border: 1px solid #CAD7E2;
    color: #202124;
    padding: 4px;
}
QMenu::item {
    background-color: transparent;
    color: #202124;
    padding: 6px 24px 6px 10px;
}
QMenu::item:selected {
    background-color: #168C91;
    color: #FFFFFF;
}
QMenu::item:disabled {
    color: #737B83;
}
QMenu::separator {
    background-color: #D8E2EB;
    height: 1px;
    margin: 4px 6px;
}
QTabWidget::pane {
    background-color: #FFFFFF;
    border: 1px solid #D8E2EB;
    border-radius: 7px;
    top: -1px;
}
QTabBar::tab {
    background-color: #E7EEF4;
    border: 1px solid #D8E2EB;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 12px;
    margin-right: 3px;
}
QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #167D83;
    font-weight: 600;
}
QProgressBar {
    background-color: #E1E9EF;
    border: none;
    border-radius: 5px;
    color: #233547;
    text-align: center;
    min-height: 14px;
}
QProgressBar::chunk {
    background-color: #20A39E;
    border-radius: 5px;
}
"""


class TranscriptionWorker(QThread):
    progress = Signal(str, float)
    succeeded = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, audio_path: str, hf_token: str, language: str | None):
        super().__init__()
        self.audio_path = audio_path
        self.hf_token = hf_token
        self.language = language

    def request_cancel(self) -> None:
        self.requestInterruption()

    def run(self) -> None:
        try:
            from transcriber import TranscriptionCancelled, transcribe_and_diarize

            result = transcribe_and_diarize(
                audio_path=self.audio_path,
                hf_token=self.hf_token,
                language=self.language,
                progress_callback=self.progress.emit,
                cancellation_callback=self.isInterruptionRequested,
            )
            self.succeeded.emit(result)
        except TranscriptionCancelled:
            self.cancelled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class GeminiWorker(QThread):
    succeeded = Signal(dict)
    failed = Signal(str, dict)

    def __init__(self, transcript: str, api_key: str):
        super().__init__()
        self.transcript = transcript
        self.api_key = api_key

    def run(self) -> None:
        try:
            suggestions = detectar_nomes_falantes(
                transcricao=self.transcript,
                api_key=self.api_key,
            )
            self.succeeded.emit(suggestions)
        except GeminiRequestError as error:
            self.failed.emit(str(error), error.sugestoes_locais)
        except Exception:
            self.failed.emit(traceback.format_exc(), {})


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(APP_STYLESHEET)
        self.setWindowTitle("Transcript Audio Text")
        self.resize(1050, 760)

        self.audio_path: Path | None = None
        self.result: dict | None = None
        self.name_edits: dict[str, QLineEdit] = {}
        self.suggestions: dict[str, str] = {}
        self.transcription_worker: TranscriptionWorker | None = None
        self.gemini_worker: GeminiWorker | None = None
        self._close_when_workers_idle = False
        self._cancel_requested = False

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        settings = QGroupBox("Configuração")
        settings_form = QFormLayout(settings)

        audio_row = QHBoxLayout()
        self.audio_label = QLabel("Nenhum arquivo selecionado")
        self.audio_label.setWordWrap(True)
        self.browse_button = QPushButton("Selecionar áudio…")
        self.browse_button.clicked.connect(self._browse_audio)
        audio_row.addWidget(self.audio_label, 1)
        audio_row.addWidget(self.browse_button)
        settings_form.addRow("Arquivo de áudio:", audio_row)

        self.language_combo = QComboBox()
        self.language_combo.addItem("Detectar automaticamente", None)
        for code, name in (
            ("pt", "Português"),
            ("en", "Inglês"),
            ("es", "Espanhol"),
            ("fr", "Francês"),
            ("de", "Alemão"),
            ("it", "Italiano"),
        ):
            self.language_combo.addItem(name, code)
        settings_form.addRow("Idioma:", self.language_combo)

        root.addWidget(settings)

        self.transcribe_button = QPushButton("Transcrever áudio")
        self.transcribe_button.setObjectName("primaryAction")
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.clicked.connect(self._start_transcription)
        self.cancel_button = QPushButton("Cancelar transcrição")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._request_transcription_cancel)
        transcription_actions = QHBoxLayout()
        transcription_actions.addWidget(self.transcribe_button, 1)
        transcription_actions.addWidget(self.cancel_button)
        root.addLayout(transcription_actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.status_label = QLabel("Selecione um arquivo de áudio para começar.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.result_tabs = QTabWidget()

        speakers_page = QWidget()
        speakers_layout = QVBoxLayout(speakers_page)
        self.speakers_group = QGroupBox("Revisão dos falantes")
        self.speakers_form = QFormLayout(self.speakers_group)
        speakers_layout.addWidget(self.speakers_group)
        speakers_layout.addWidget(QLabel("Transcrição por falante:"))
        self.speaker_text = QTextEdit()
        self.speaker_text.setReadOnly(True)
        speakers_layout.addWidget(self.speaker_text, 1)
        self.result_tabs.addTab(speakers_page, "Por falante")

        self.timestamp_text = QTextEdit()
        self.timestamp_text.setReadOnly(True)
        self.result_tabs.addTab(self.timestamp_text, "Com marcas de tempo")

        self.plain_text = QTextEdit()
        self.plain_text.setReadOnly(True)
        self.result_tabs.addTab(self.plain_text, "Texto puro")

        root.addWidget(self.result_tabs, 1)

        gemini_group = QGroupBox("Sugestões de nomes (opcional)")
        gemini_layout = QVBoxLayout(gemini_group)
        gemini_key_row = QHBoxLayout()
        self.gemini_key_edit = QLineEdit()
        self.gemini_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_edit.setPlaceholderText("API key própria do Google AI Studio")
        self.suggest_button = QPushButton("Sugerir nomes com Gemini")
        self.suggest_button.setEnabled(False)
        self.suggest_button.clicked.connect(self._confirm_and_suggest)
        gemini_key_row.addWidget(self.gemini_key_edit, 1)
        gemini_key_row.addWidget(self.suggest_button)
        gemini_layout.addLayout(gemini_key_row)
        self.suggestion_status = QLabel(
            "A chave não será salva. A transcrição só será enviada ao Google "
            "se você solicitar sugestões."
        )
        self.suggestion_status.setWordWrap(True)
        gemini_layout.addWidget(self.suggestion_status)
        root.addWidget(gemini_group)

        export_row = QHBoxLayout()
        self.docx_button = QPushButton("Baixar por Falante (DOCX)")
        self.docx_button.setObjectName("exportAction")
        self.timestamps_button = QPushButton("Baixar Marcas de Tempo (TXT)")
        self.timestamps_button.setObjectName("exportAction")
        self.docx_button.clicked.connect(self._save_docx)
        self.timestamps_button.clicked.connect(self._save_timestamps)
        for button in (self.docx_button, self.timestamps_button):
            button.setEnabled(False)
            export_row.addWidget(button)
        root.addLayout(export_row)

        self.setCentralWidget(central)

    def _browse_audio(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar arquivo de áudio",
            "",
            f"{AUDIO_EXTENSIONS};;Todos os arquivos (*)",
        )
        if not filename:
            return
        self.audio_path = Path(filename)
        self.audio_label.setText(filename)
        self.transcribe_button.setEnabled(True)
        self._clear_result()
        self._set_status("Arquivo selecionado. Inicie a transcrição.")

    def _start_transcription(self) -> None:
        if self.audio_path is None:
            return
        hf_token = HF_TOKEN_FOR_TEST.strip() or os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            QMessageBox.warning(
                self,
                "Configuração necessária",
                "Defina a variável de ambiente HF_TOKEN com seu token do Hugging Face "
                "e reinicie o aplicativo.",
            )
            return

        self._clear_result()
        self._cancel_requested = False
        self._set_busy(True)
        self.progress.setValue(0)
        self._set_status("Iniciando transcrição…")
        self.transcription_worker = TranscriptionWorker(
            str(self.audio_path),
            hf_token,
            self.language_combo.currentData(),
        )
        self.transcription_worker.progress.connect(self._update_progress)
        self.transcription_worker.succeeded.connect(self._transcription_succeeded)
        self.transcription_worker.failed.connect(self._worker_failed)
        self.transcription_worker.cancelled.connect(self._transcription_cancelled)
        self.transcription_worker.finished.connect(lambda: self._set_busy(False))
        self.transcription_worker.finished.connect(self._finish_pending_close)
        self.transcription_worker.start()

    def _set_status(self, message: str, *, bold: bool = False) -> None:
        status_font = self.status_label.font()
        status_font.setBold(bold)
        self.status_label.setFont(status_font)
        self.status_label.setText(message)

    def _request_transcription_cancel(self) -> None:
        worker = self.transcription_worker
        if worker is None or not worker.isRunning():
            return
        worker.request_cancel()
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self._set_status(
            "Cancelamento solicitado. A etapa atual será concluída antes de parar."
        )

    def _transcription_cancelled(self) -> None:
        self._cancel_requested = False
        self._set_status("Transcrição cancelada; nenhum resultado parcial foi salvo.")
        self.progress.setValue(0)

    def _finish_pending_close(self) -> None:
        if not self._close_when_workers_idle:
            return
        if any(
            worker is not None and worker.isRunning()
            for worker in (self.transcription_worker, self.gemini_worker)
        ):
            return
        self._close_when_workers_idle = False
        self.close()

    def _update_progress(self, message: str, amount: float) -> None:
        if self._cancel_requested:
            return
        self._set_status(message)
        self.progress.setValue(max(0, min(100, round(amount * 100))))

    def _transcription_succeeded(self, result: dict) -> None:
        self.result = result
        self.suggestions = {}
        self._build_speaker_editor()
        local_suggestions = sugerir_nomes_locais(result["full_text"])
        self._apply_suggestions(local_suggestions)
        self._refresh_outputs()
        self.suggest_button.setEnabled(True)
        self.docx_button.setEnabled(True)
        self.timestamps_button.setEnabled(True)
        if local_suggestions:
            self.suggestion_status.setText(
                f"Foram encontradas localmente {len(local_suggestions)} sugestão(ões) "
                "explícitas, sem enviar dados. Revise ou ajuste os nomes nos campos "
                "da seção “Revisão dos falantes”."
            )
        else:
            self.suggestion_status.setText(
                "Nenhuma apresentação explícita foi identificada localmente. "
                "Você pode preencher os nomes manualmente ou solicitar sugestões ao Gemini."
            )
        self._set_status(
            f"Concluído. Idioma: {result['language']}. "
            f"Falantes identificados: {len(result['speakers'])}.",
            bold=True,
        )
        self.progress.setValue(100)

    def _worker_failed(self, details: str) -> None:
        self._set_status("A operação falhou.")
        QMessageBox.critical(self, "Erro no processamento", details)

    def _build_speaker_editor(self) -> None:
        while self.speakers_form.rowCount():
            self.speakers_form.removeRow(0)
        self.name_edits.clear()
        if not self.result:
            return

        segments = self.result["segments"]
        for speaker in self.result["speakers"]:
            first_utterance = next(
                (segment["text"] for segment in segments if segment["speaker"] == speaker),
                "",
            )
            edit = QLineEdit()
            edit.setPlaceholderText(speaker)
            edit.textChanged.connect(self._refresh_outputs)
            self.name_edits[speaker] = edit
            self.speakers_form.addRow(
                f"{speaker} — {first_utterance[:120]}",
                edit,
            )

    def _speaker_map(self) -> dict[str, str]:
        return {
            speaker: edit.text().strip()
            for speaker, edit in self.name_edits.items()
        }

    def _display_name(self, speaker: str) -> str:
        return self._speaker_map().get(speaker, "") or speaker

    def _refresh_outputs(self, *_args) -> None:
        if not self.result:
            return
        mapped = format_transcript_by_speaker(
            self.result["full_text"],
            self._speaker_map(),
        )
        self.speaker_text.setPlainText(mapped)
        self.plain_text.setPlainText(
            "\n".join(
                f"{self._display_name(segment['speaker'])}: {segment['text']}"
                for segment in self.result["segments"]
            )
        )
        self.timestamp_text.setPlainText(
            "\n".join(
                f"[{segment['start_hms']} → {segment['end_hms']}] "
                f"{self._display_name(segment['speaker'])}: {segment['text']}"
                for segment in self.result["segments"]
            )
        )

    def _confirm_and_suggest(self) -> None:
        if not self.result:
            return
        api_key = self.gemini_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(
                self,
                "API key necessária",
                "Informe sua API key do Google AI Studio para solicitar sugestões.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Enviar transcrição ao Gemini?",
            "Para sugerir nomes, o texto da transcrição (até 12.000 caracteres) "
            "será enviado à API do Google usando sua chave. Deseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.suggest_button.setEnabled(False)
        self.suggestion_status.setText("Consultando o Gemini…")
        self.gemini_worker = GeminiWorker(
            self.result["full_text"],
            api_key,
        )
        self.gemini_worker.succeeded.connect(self._suggestions_succeeded)
        self.gemini_worker.failed.connect(self._suggestions_failed)
        self.gemini_worker.finished.connect(
            lambda: self.suggest_button.setEnabled(self.result is not None)
        )
        self.gemini_worker.finished.connect(self._finish_pending_close)
        self.gemini_worker.start()

    def _apply_suggestions(self, suggestions: dict[str, str]) -> int:
        previous_suggestions = self.suggestions
        self.suggestions = suggestions
        count = 0
        for speaker, name in suggestions.items():
            edit = self.name_edits.get(speaker)
            current_name = edit.text().strip() if edit else ""
            if name and edit and (
                not current_name or current_name == previous_suggestions.get(speaker)
            ):
                edit.setText(name)
                count += 1
        return count

    def _suggestions_succeeded(self, suggestions: dict) -> None:
        count = self._apply_suggestions(suggestions)
        found = sum(bool(name) for name in suggestions.values())
        self.suggestion_status.setText(
            f"Gemini encontrou {found} sugestão(ões); {count} aplicada(s). "
            "Confira os nomes antes de exportar; alterações manuais foram preservadas."
        )

    def _suggestions_failed(self, message: str, local_suggestions: dict) -> None:
        if local_suggestions:
            self._apply_suggestions(local_suggestions)
            count = sum(bool(name) for name in local_suggestions.values())
            self.suggestion_status.setText(
                f"{message}\nAs sugestões locais ({count}) continuam nos campos editáveis. "
                "Confira ou ajuste os nomes manualmente."
            )
            return
        self.suggestion_status.setText(f"Erro ao consultar o Gemini: {message}")
        QMessageBox.critical(self, "Erro ao consultar o Gemini", message)

    def _save_file(self, extension: str, file_filter: str, content: bytes | str) -> None:
        if not self.audio_path:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar transcrição",
            str(self.audio_path.with_name(f"{self.audio_path.stem}{extension}")),
            file_filter,
        )
        if not filename:
            return
        try:
            mode = "wb" if isinstance(content, bytes) else "w"
            kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
            with open(filename, mode, **kwargs) as output:
                output.write(content)
        except OSError as error:
            QMessageBox.critical(self, "Erro ao salvar", str(error))
            return
        self._set_status(f"Arquivo salvo: {filename}")

    def _save_docx(self) -> None:
        self._save_file(
            "_transcricao.docx",
            "Documento Word (*.docx)",
            transcript_to_docx(self.speaker_text.toPlainText()),
        )

    def _save_timestamps(self) -> None:
        if not self.result:
            return
        self._save_file(
            "_timestamps.txt",
            "Arquivo de texto (*.txt)",
            self.timestamp_text.toPlainText(),
        )

    def _clear_result(self) -> None:
        self.result = None
        self.suggestions = {}
        self.name_edits.clear()
        while self.speakers_form.rowCount():
            self.speakers_form.removeRow(0)
        for widget in (self.speaker_text, self.timestamp_text, self.plain_text):
            widget.clear()
        self.suggest_button.setEnabled(False)
        self.docx_button.setEnabled(False)
        self.timestamps_button.setEnabled(False)
        self.suggestion_status.setText(
            "A chave não será salva. A transcrição só será enviada ao Google "
            "se você solicitar sugestões."
        )

    def _set_busy(self, busy: bool) -> None:
        self.transcribe_button.setEnabled(not busy and self.audio_path is not None)
        self.cancel_button.setEnabled(busy)
        self.browse_button.setEnabled(not busy)
        self.language_combo.setEnabled(not busy)

    def closeEvent(self, event) -> None:
        active_transcription = (
            self.transcription_worker is not None
            and self.transcription_worker.isRunning()
        )
        active_gemini = (
            self.gemini_worker is not None and self.gemini_worker.isRunning()
        )
        if not active_transcription and not active_gemini:
            event.accept()
            return

        if self._close_when_workers_idle:
            event.ignore()
            return

        answer = QMessageBox.question(
            self,
            "Processamento em andamento",
            "Deseja cancelar a transcrição e fechar quando a etapa atual terminar?"
            if active_transcription
            else "A solicitação ao Gemini ainda está em andamento. Fechar quando ela terminar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        self._close_when_workers_idle = True
        if active_transcription:
            self._request_transcription_cancel()
        else:
            self._set_status("Fechando após a resposta do Gemini…")
        event.ignore()


def main() -> int:
    if "--check-runtime" in sys.argv:
        from transcriber import transcribe_and_diarize  # noqa: F401

        print("WhisperX runtime imports successfully.")
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("Transcrição de áudio")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
