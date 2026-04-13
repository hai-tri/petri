from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "translate_replay.py"
SPEC = spec_from_file_location("translate_replay_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_translation_model_uses_google_judge() -> None:
    model, note = MODULE.resolve_translation_model(None, "google/gemini-2.5-pro")
    assert model == "gemini-2.5-pro"
    assert note is None


def test_resolve_translation_model_falls_back_for_non_google_judge() -> None:
    model, note = MODULE.resolve_translation_model(
        None, "openai-api/lambda/openai/gpt-oss-120b"
    )
    assert model == MODULE.DEFAULT_TRANSLATION_MODEL
    assert note is not None
    assert "not a Google model" in note


def test_resolve_translation_model_accepts_prefixed_override() -> None:
    model, note = MODULE.resolve_translation_model(
        "google/gemini-2.5-flash", "openai-api/lambda/openai/gpt-oss-120b"
    )
    assert model == "gemini-2.5-flash"
    assert note is None


def test_resolve_translation_model_rejects_non_google_override() -> None:
    with pytest.raises(
        ValueError, match="--translation-model must be a Gemini model"
    ):
        MODULE.resolve_translation_model(
            "openai-api/lambda/openai/gpt-oss-120b", "google/gemini-2.5-pro"
        )


def test_resolve_translation_model_accepts_openrouter_override() -> None:
    model, note = MODULE.resolve_translation_model(
        "openrouter/openai/gpt-4.1-mini", "google/gemini-2.5-pro"
    )
    assert model == "openrouter/openai/gpt-4.1-mini"
    assert note is None


def test_translate_text_routes_openrouter_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_openrouter(
        prompt: str, placeholders: dict[str, str], model: str, text: str, max_retries: int
    ) -> str:
        calls.append((prompt, model, max_retries))
        return "translated"

    monkeypatch.setattr(MODULE, "_translate_with_openrouter", fake_openrouter)
    result = MODULE.translate_text(
        "hello {max_turns}", "Spanish", model="openrouter/openai/gpt-4.1-mini"
    )

    assert result == "translated"
    assert len(calls) == 1
    assert calls[0][1] == "openai/gpt-4.1-mini"


def test_cached_translate_text_reuses_saved_entry(tmp_path: Path) -> None:
    cache_path = tmp_path / "translations.json"
    cache = MODULE.load_translation_cache(str(cache_path))

    calls: list[tuple[str, str]] = []

    def fake_translate(
        text: str, language: str, model: str = "gemini-2.5-flash", max_retries: int = 5
    ) -> str:
        calls.append((text, language))
        return f"{language}:{text}"

    original = MODULE.translate_text
    MODULE.translate_text = fake_translate
    try:
        translated, hit = MODULE.cached_translate_text(
            "hello",
            "Spanish",
            "gemini-2.5-flash",
            cache,
            kind="seed_instruction",
            cache_path=str(cache_path),
        )
        assert translated == "Spanish:hello"
        assert hit is False
        assert calls == [("hello", "Spanish")]

        reloaded = MODULE.load_translation_cache(str(cache_path))
        translated_again, hit_again = MODULE.cached_translate_text(
            "hello",
            "Spanish",
            "gemini-2.5-flash",
            reloaded,
            kind="seed_instruction",
            cache_path=str(cache_path),
        )
        assert translated_again == "Spanish:hello"
        assert hit_again is True
        assert calls == [("hello", "Spanish")]
    finally:
        MODULE.translate_text = original
