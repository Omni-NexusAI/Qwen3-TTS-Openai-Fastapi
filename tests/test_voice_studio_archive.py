"""Focused, dependency-light contracts for the archival Gradio Voice Studio."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_studio():
    """Load helper-level Studio code without installing the full Gradio stack."""
    gradio = types.ModuleType("gradio")
    gradio.Blocks = object
    gradio.themes = types.SimpleNamespace(Soft=object)
    httpx = types.ModuleType("httpx")
    httpx.Client = object
    original_gradio = sys.modules.get("gradio")
    original_httpx = sys.modules.get("httpx")
    sys.modules["gradio"] = gradio
    sys.modules["httpx"] = httpx
    try:
        spec = importlib.util.spec_from_file_location("archive_voice_studio", ROOT / "gradio_voice_studio.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_gradio is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = original_gradio
        if original_httpx is None:
            sys.modules.pop("httpx", None)
        else:
            sys.modules["httpx"] = original_httpx


def test_profile_ids_cannot_escape_voice_library(tmp_path):
    studio = _load_studio()
    for profile_id in ("../outside", "a/b", "", "with space"):
        with pytest.raises(ValueError, match="Profile id is invalid"):
            studio.profile_dir(tmp_path, profile_id)


def test_profile_metadata_must_match_selected_profile(tmp_path):
    studio = _load_studio()
    target = studio.profile_dir(tmp_path, "valid-profile")
    target.mkdir(parents=True)
    (target / "meta.json").write_text(
        json.dumps({
            "profile_id": "other-profile",
            "name": "Archive test",
            "task_type": "Base",
            "created_at": "2026-01-01T00:00:00Z",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match"):
        studio.load_profile(tmp_path, "valid-profile")


def test_streaming_timing_is_truthful_when_server_omits_terminal_metrics():
    studio = _load_studio()
    rendered = studio.timing_markdown({"chunk_count": 2, "completed": False}, delivery_mode="Streaming")
    assert "ended without final timing metadata" in rendered
    assert "not reported" in rendered
    assert "browser-incremental playback" in rendered


def test_archive_studio_has_no_hard_coded_debug_telemetry_endpoint():
    source = (ROOT / "gradio_voice_studio.py").read_text(encoding="utf-8")
    assert "127.0.0.1:7852" not in source
    assert "_debug_log" not in source
