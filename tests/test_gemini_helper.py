import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gemini_helper import (
    GeminiRequestError,
    _extrair_nomes_heuristicos,
    _normalizar_chave_api,
    detectar_nomes_falantes,
    sugerir_nomes_locais,
)


class SpeakerNameHeuristicsTests(unittest.TestCase):
    def test_detects_names_in_multiple_languages_and_repeated_turns(self):
        transcript = (
            "[SPEAKER_00] Meu nome é João da Silva. Eu sou o Dr. João da Silva.\n\n"
            "[SPEAKER_01] My name is Ana. Thanks for having me.\n\n"
            "[SPEAKER_00] Sou João da Silva e vou começar.\n\n"
            "[SPEAKER_02] Me llamo Carlos."
        )

        self.assertEqual(
            _extrair_nomes_heuristicos(transcript),
            {
                "SPEAKER_00": "João da Silva",
                "SPEAKER_01": "Ana",
                "SPEAKER_02": "Carlos",
            },
        )

    def test_omits_conflicting_names_instead_of_guessing(self):
        transcript = (
            "[SPEAKER_00] Meu nome é Ana.\n\n"
            "[SPEAKER_00] Na verdade, me chamo Beatriz."
        )

        self.assertEqual(_extrair_nomes_heuristicos(transcript), {})

    def test_local_suggestions_are_available_without_api_call(self):
        self.assertEqual(
            sugerir_nomes_locais(
                "[SPEAKER_00] Eu sou a Maria, prazer.\n\n"
                "[SPEAKER_01] Ainda não me apresentei."
            ),
            {"SPEAKER_00": "Maria"},
        )

    def test_rejects_oauth_access_tokens_as_api_keys(self):
        for token in ("ya29.access-token", "Bearer ya29.access-token", "1//oauth-token"):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "token OAuth"):
                    _normalizar_chave_api(token)

    def test_accepts_ai_studio_key_without_rewriting_it(self):
        api_key = "AIzaExampleTestKey"

        self.assertEqual(_normalizar_chave_api(api_key), api_key)


class GeminiNameSuggestionTests(unittest.TestCase):
    @patch("gemini_helper.genai.Client")
    def test_merges_local_suggestions_when_model_omits_or_blanks_names(self, client_factory):
        client = client_factory.return_value
        client.chats.create.return_value.send_message.return_value = SimpleNamespace(
            text=json.dumps({"SPEAKER_00": "", "SPEAKER_99": "Invented"})
        )
        transcript = (
            "[SPEAKER_00] Meu nome é João.\n\n"
            "[SPEAKER_01] My name is Ana."
        )

        result = detectar_nomes_falantes(transcript, "AIzaExampleTestKey")

        self.assertEqual(result, {"SPEAKER_00": "João", "SPEAKER_01": "Ana"})
        client_factory.assert_called_once_with(api_key="AIzaExampleTestKey")
        client.chats.create.assert_called_once()
        client.chats.create.return_value.send_message.assert_called_once()

    @patch("gemini_helper.genai.Client")
    def test_surfaces_authentication_failure_with_local_suggestions(self, client_factory):
        client_factory.return_value.chats.create.return_value.send_message.side_effect = RuntimeError(
            "401 UNAUTHENTICATED ACCESS_TOKEN_TYPE_UNSUPPORTED"
        )

        with self.assertRaises(GeminiRequestError) as raised:
            detectar_nomes_falantes(
                "[SPEAKER_00] Meu nome é João.",
                "AIzaExampleTestKey",
            )

        self.assertIn("API key ativa do Google AI Studio", str(raised.exception))
        self.assertEqual(raised.exception.sugestoes_locais, {"SPEAKER_00": "João"})

    @patch("gemini_helper.genai.Client")
    def test_does_not_hide_other_api_failures_when_local_suggestions_exist(self, client_factory):
        client_factory.return_value.chats.create.return_value.send_message.side_effect = RuntimeError(
            "service unavailable"
        )

        with self.assertRaisesRegex(GeminiRequestError, "service unavailable"):
            detectar_nomes_falantes(
                "[SPEAKER_00] Meu nome é João.",
                "AIzaExampleTestKey",
            )

    @patch("gemini_helper.genai.Client")
    def test_explains_unavailable_model_and_api_recommendation(self, client_factory):
        client_factory.return_value.chats.create.return_value.send_message.side_effect = RuntimeError(
            "404 NOT_FOUND: model gemini-2.5-flash is unavailable; "
            "Please update your code to use models/gemini-3.8-flash"
        )

        with self.assertRaises(GeminiRequestError) as raised:
            detectar_nomes_falantes(
                "[SPEAKER_00] Meu nome é João.",
                "AIzaExampleTestKey",
                model="gemini-2.5-flash",
            )

        self.assertIn("não encontrou o modelo 'gemini-2.5-flash' (404)", str(raised.exception))
        self.assertIn("recomenda 'gemini-3.8-flash'", str(raised.exception))
        self.assertEqual(raised.exception.sugestoes_locais, {"SPEAKER_00": "João"})

    @patch("gemini_helper.genai.Client")
    def test_summarizes_temporary_service_unavailability(self, client_factory):
        client_factory.return_value.chats.create.return_value.send_message.side_effect = RuntimeError(
            "503 UNAVAILABLE: high demand"
        )

        with self.assertRaises(GeminiRequestError) as raised:
            detectar_nomes_falantes("[SPEAKER_00] Bom dia.", "AIzaExampleTestKey")

        self.assertIn("temporariamente indisponível", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
