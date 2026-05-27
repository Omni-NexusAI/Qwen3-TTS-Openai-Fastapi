# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import asyncio
import base64
import inspect
import io
import json
import logging
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

try:
    # Optional vLLM proxy bridge; not required for optimized/backend-only setups.
    from ..vllm_bridge import proxy_vllm_speech, proxy_vllm_speech_stream_framed
except ImportError:  # pragma: no cover - best-effort fallback for non-vLLM environments
    proxy_vllm_speech = None
    proxy_vllm_speech_stream_framed = None
from ..structures.schemas import (
    OpenAISpeechRequest,
    BackendModelSwitchRequest,
    ModelInfo,
    VoiceInfo,
    VoiceCloneRequest,
    VoiceCloneCapabilities,
    StreamingVoiceCloneRequest,
    TimingInfo,
)
from ..services.text_processing import normalize_text
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Voice library: saved voice profiles used via the "clone:ProfileName" voice prefix
VOICE_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()
_ref_audio_cache: dict = {}


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": "35990f",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        httpx.post(
            "http://127.0.0.1:7852/ingest/d65a6209-8a13-4b06-b1f6-8a0a5aeed5c8",
            headers={"Content-Type": "application/json", "X-Debug-Session-Id": "35990f"},
            json=payload,
            timeout=1.0,
        )
    except Exception:
        pass


async def _ensure_vllm_ready(vllm_url: str, timeout_s: float = 2.0) -> None:
    """Fail fast with 503 when the external vllm serve is not reachable yet."""
    base = (vllm_url or "").rstrip("/")
    if not base:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "vllm_unconfigured",
                "message": "VLLM_SERVE_URL is not set.",
                "type": "server_error",
            },
        )
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(f"{base}/health")
            if r.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "vllm_not_ready",
                        "message": f"vLLM is starting (health={r.status_code}). Try again shortly.",
                        "type": "server_error",
                    },
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "vllm_unreachable",
                "message": f"vLLM is not reachable yet: {exc}",
                "type": "server_error",
            },
        )


def _load_voice_profile(name_or_id: str) -> dict:
    """Load a voice profile by name or profile_id from the voice library."""
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if not profiles_dir.exists():
        raise ValueError(f"Voice library not found: {profiles_dir}")
    for child in sorted(profiles_dir.iterdir()):
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("profile_id") == name_or_id or meta.get("name", "").lower() == name_or_id.lower():
            ref_filename = meta.get("ref_audio_filename", "")
            if not ref_filename:
                raise ValueError(f"Profile '{name_or_id}' has no reference audio filename")
            ref_path = child / ref_filename
            if not ref_path.exists():
                raise ValueError(f"Reference audio missing: {ref_path}")
            return {
                "ref_audio_path": str(ref_path),
                "ref_text": meta.get("ref_text", ""),
                "x_vector_only_mode": meta.get("x_vector_only_mode", False),
                "language": meta.get("language", "Auto"),
                "name": meta.get("name", name_or_id),
            }
    raise ValueError(f"Voice profile not found: '{name_or_id}'")


def _to_audio_data_url(audio_b64: str, filename: Optional[str] = None) -> str:
    """Ensure ref_audio is a data URL expected by vLLM-Omni 0.17."""
    if audio_b64.startswith("data:"):
        return audio_b64
    mime = "audio/wav"
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            mime = guessed
    return f"data:{mime};base64,{audio_b64}"


def _method_accepts_kwarg(method, kwarg: str) -> bool:
    """Return True if a callable accepts a given keyword argument."""
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    if kwarg in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Sophia",
    "nova": "Isabella",
    "onyx": "Evan",
    "shimmer": "Lily",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """
    Extract language from model name if it has a language suffix.
    
    Args:
        model_name: Model name (e.g., "tts-1-es", "tts-1-hd-fr")
    
    Returns:
        Language name if suffix found, None otherwise
    """
    # Check if model ends with a language code
    # Only extract language if the model follows the expected pattern
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            # Verify it's a valid language-specific model variant
            # Should be either tts-1-{lang} or tts-1-hd-{lang}
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend
    
    backend = get_backend()
    
    if not backend.is_ready():
        await initialize_backend()
    
    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    # Check OpenAI voice mapping first
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    # Otherwise use the voice name directly
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text using the configured TTS backend.
    
    Args:
        text: The text to synthesize
        voice: Voice name to use
        language: Language code
        instruct: Optional instruction for voice style
        speed: Speech speed multiplier
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    backend = await get_tts_backend()

    # Check custom voice BEFORE applying OpenAI alias mapping,
    # so custom voices with OpenAI alias names remain accessible.
    if backend.is_custom_voice(voice):
        try:
            audio, sr = await backend.generate_speech_with_custom_voice(
                text=text,
                voice=voice,
                language=language,
                speed=speed,
            )
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Speech generation failed: {e}")

    # Map voice name (OpenAI aliases to internal names)
    voice_name = get_voice_name(voice)
    
    # Generate speech using the backend
    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )
        
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """
    OpenAI-compatible endpoint for text-to-speech.
    Supports voice library via voice="clone:ProfileName" and streaming when backend supports it.
    """
    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )

    normalized_text = normalize_text(request.input, request.normalization_options)
    if not normalized_text.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_input",
                "message": "Input text is empty after normalization",
                "type": "invalid_request_error",
            },
        )

    model_language = extract_language_from_model(request.model)
    language = model_language if model_language else (request.language or "Auto")

    # ---------------------------------------------------------------------
    # vLLM-Omni proxy mode (when vllm serve is running in-container)
    # ---------------------------------------------------------------------
    if os.getenv("TTS_BACKEND", "").lower() in ("vllm_omni", "vllm-omni", "vllm") and os.getenv("VLLM_SERVE_URL"):
        vllm_url = os.environ["VLLM_SERVE_URL"]
        # clone:ProfileName -> attach ref_audio/ref_text and set Base task
        if request.voice.lower().startswith("clone:"):
            profile_name = request.voice[len("clone:"):].strip()
            if not profile_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_voice",
                        "message": "The 'clone:' prefix requires a profile name, e.g. voice='clone:MyVoice'",
                        "type": "invalid_request_error",
                    },
                )
            try:
                profile = _load_voice_profile(profile_name)
            except ValueError as exc:
                raise HTTPException(
                    status_code=404,
                    detail={"error": "profile_not_found", "message": str(exc), "type": "invalid_request_error"},
                )
            if not profile["x_vector_only_mode"] and not profile["ref_text"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_ref_text",
                        "message": f"Profile '{profile['name']}' is configured for ICL mode but has no ref_text.",
                        "type": "invalid_request_error",
                    },
                )
            canonical_key = profile["name"].lower()
            ref_path = profile["ref_audio_path"]
            if canonical_key not in _ref_audio_cache:
                try:
                    ref_audio_np, ref_sr = sf.read(ref_path)
                    if len(ref_audio_np.shape) > 1:
                        ref_audio_np = ref_audio_np.mean(axis=1)
                    ref_audio_np = ref_audio_np.astype(np.float32)
                    _ref_audio_cache[canonical_key] = (ref_audio_np, ref_sr)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "audio_processing_error",
                            "message": f"Failed to load reference audio for profile '{profile['name']}': {exc}",
                            "type": "invalid_request_error",
                        },
                    )
            # vLLM online serving accepts ref_audio as URL or base64 data; we provide base64 bytes.
            # We re-encode from file bytes to avoid float->wav conversion.
            with open(ref_path, "rb") as f:
                ref_b64 = base64.b64encode(f.read()).decode("utf-8")
            task_type = "Base"
            payload = {
                "input": normalized_text,
                "voice": "vivian",
                "language": language if language != "Auto" else profile["language"],
                "instructions": request.instruct or "",
                "task_type": task_type,
                "ref_audio": _to_audio_data_url(ref_b64, ref_path),
                "ref_text": profile["ref_text"] or "",
                "x_vector_only_mode": bool(profile["x_vector_only_mode"]),
                "stream": bool(request.stream),
                "response_format": request.response_format,
            }
        else:
            backend = await get_tts_backend()
            if getattr(backend, "get_model_type", lambda: "unknown")() == "base":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "base_model_requires_clone",
                        "message": "Current model is Base (clone-only). Use voice='clone:ProfileName' or /v1/audio/voice-clone endpoints.",
                        "type": "invalid_request_error",
                    },
                )
            payload = {
                "input": normalized_text,
                "voice": request.voice,
                "language": language,
                "instructions": request.instruct or "",
                "task_type": "CustomVoice",
                "stream": bool(request.stream),
                "response_format": request.response_format,
                "speed": request.speed,
            }

        if payload.get("stream"):
            await _ensure_vllm_ready(vllm_url)
            if payload.get("response_format") != "pcm":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_format_for_streaming",
                        "message": "vLLM streaming requires response_format='pcm'.",
                        "type": "invalid_request_error",
                    },
                )
            if payload.get("speed", 1.0) != 1.0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_speed_for_streaming",
                        "message": "vLLM streaming does not support speed adjustment. Use speed=1.0.",
                        "type": "invalid_request_error",
                    },
                )
            return StreamingResponse(
                proxy_vllm_speech_stream_framed(vllm_url, payload, timeout_s=300.0),
                media_type="application/octet-stream",
                headers={"Cache-Control": "no-cache", "X-Streaming": "true"},
            )

        # non-streaming
        await _ensure_vllm_ready(vllm_url)
        content = await proxy_vllm_speech(vllm_url, payload, timeout_s=300.0)
        return Response(
            content=content,
            media_type=get_content_type(request.response_format),
            headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}", "Cache-Control": "no-cache"},
        )

    # Voice library: "clone:ProfileName" -> load profile + voice clone (stream or non-stream)
    if request.voice.lower().startswith("clone:"):
        profile_name = request.voice[len("clone:"):].strip()
        if not profile_name:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_voice",
                    "message": "The 'clone:' prefix requires a profile name, e.g. voice='clone:MyVoice'",
                    "type": "invalid_request_error",
                },
            )
        try:
            profile = _load_voice_profile(profile_name)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail={"error": "profile_not_found", "message": str(exc), "type": "invalid_request_error"},
            )
        backend = await get_tts_backend()
        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice library cloning requires a Base model and the optimized backend or a backend that supports voice cloning.",
                    "type": "invalid_request_error",
                },
            )
        if not profile["x_vector_only_mode"] and not profile["ref_text"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": f"Profile '{profile['name']}' is configured for ICL mode but has no ref_text. Add a transcript to meta.json or set x_vector_only_mode=true.",
                    "type": "invalid_request_error",
                },
            )
        canonical_key = profile["name"].lower()
        ref_path = profile["ref_audio_path"]
        if canonical_key not in _ref_audio_cache:
            try:
                ref_audio_np, ref_sr = sf.read(ref_path)
                if len(ref_audio_np.shape) > 1:
                    ref_audio_np = ref_audio_np.mean(axis=1)
                ref_audio_np = ref_audio_np.astype(np.float32)
                _ref_audio_cache[canonical_key] = (ref_audio_np, ref_sr)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "audio_processing_error",
                        "message": f"Failed to load reference audio for profile '{profile['name']}': {exc}",
                        "type": "invalid_request_error",
                    },
                )
        ref_audio_np, ref_sr = _ref_audio_cache[canonical_key]
        clone_lang = language if language != "Auto" else profile["language"]

        if request.stream and hasattr(backend, "generate_voice_clone_streaming"):
            if request.response_format not in ("pcm", "wav"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_format_for_streaming",
                        "message": "Real-time streaming only supports response_format 'pcm'. Use stream=false for other formats.",
                        "type": "invalid_request_error",
                    },
                )
            fmt = "pcm"
            content_type = get_content_type(fmt)
            clone_kwargs = {
                "text": normalized_text,
                "ref_audio": ref_audio_np,
                "ref_audio_sr": ref_sr,
                "ref_text": profile["ref_text"] or None,
                "language": clone_lang,
                "x_vector_only_mode": profile["x_vector_only_mode"],
            }
            if _method_accepts_kwarg(backend.generate_voice_clone_streaming, "cache_key"):
                clone_kwargs["cache_key"] = canonical_key

            async def _clone_stream():
                gen_start = time.time()
                first_logged = False
                total_samples = 0
                sample_rate = 24000
                async for pcm_chunk, sr in backend.generate_voice_clone_streaming(**clone_kwargs):
                    if pcm_chunk is not None and len(pcm_chunk) > 0:
                        if not first_logged:
                            logger.info(f"Voice clone stream TTFB: {time.time() - gen_start:.3f}s")
                            first_logged = True
                        total_samples += len(pcm_chunk)
                        sample_rate = sr
                        yield encode_audio(pcm_chunk, fmt, sr)
                    await asyncio.sleep(0)

            return StreamingResponse(
                _clone_stream(),
                media_type=content_type,
                headers={"Content-Disposition": "inline; filename=speech.pcm", "Cache-Control": "no-cache"},
            )
        # Non-streaming clone
        try:
            clone_kwargs = {
                "text": normalized_text,
                "ref_audio": ref_audio_np,
                "ref_audio_sr": ref_sr,
                "ref_text": profile["ref_text"] or None,
                "language": clone_lang,
                "x_vector_only_mode": profile["x_vector_only_mode"],
                "speed": request.speed,
            }
            if _method_accepts_kwarg(backend.generate_voice_clone, "cache_key"):
                clone_kwargs["cache_key"] = canonical_key
            audio, sample_rate = await backend.generate_voice_clone(**clone_kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": "processing_error", "message": str(e), "type": "server_error"})
        audio_bytes = await asyncio.to_thread(encode_audio, audio, request.response_format, sample_rate)
        return Response(
            content=audio_bytes,
            media_type=get_content_type(request.response_format),
            headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}", "Cache-Control": "no-cache"},
        )

    # Built-in voice streaming (when backend supports generate_speech_streaming)
    if request.stream:
        backend = await get_tts_backend()
        if hasattr(backend, "generate_speech_streaming"):
            if request.response_format not in ("pcm", "wav"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_format_for_streaming",
                        "message": "Real-time streaming only supports response_format 'pcm'. Use stream=false for other formats.",
                        "type": "invalid_request_error",
                    },
                )
            fmt = "pcm"
            content_type = get_content_type(fmt)
            voice_name = get_voice_name(request.voice)

            async def _speech_stream():
                gen_start = time.time()
                first_logged = False
                async for pcm_chunk, sr in backend.generate_speech_streaming(
                    text=normalized_text,
                    voice=voice_name,
                    language=language,
                    instruct=request.instruct,
                    speed=request.speed,
                ):
                    if pcm_chunk is not None and len(pcm_chunk) > 0:
                        if not first_logged:
                            logger.info(f"TTS stream TTFB: {time.time() - gen_start:.3f}s")
                            first_logged = True
                        yield encode_audio(pcm_chunk, fmt, sr)
                    await asyncio.sleep(0)

            return StreamingResponse(
                _speech_stream(),
                media_type=content_type,
                headers={"Content-Disposition": "inline; filename=speech.pcm", "Cache-Control": "no-cache"},
            )

    # Non-streaming (or backends without streaming)
    try:
        audio, sample_rate = await generate_speech(
            text=normalized_text,
            voice=request.voice,
            language=language,
            instruct=request.instruct,
            speed=request.speed,
        )
        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        return Response(
            content=audio_bytes,
            media_type=get_content_type(request.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "processing_error", "message": str(e), "type": "server_error"},
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/backend/models")
async def get_backend_models():
    """List available backend models and current model (optimized backend only)."""
    from ..backends import get_backend

    backend = get_backend()
    if not hasattr(backend, "get_available_models") or not hasattr(backend, "get_current_model_key"):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_supported",
                "message": "This backend does not support model listing or switching.",
                "type": "invalid_request_error",
            },
        )
    available = backend.get_available_models()
    current = backend.get_current_model_key()
    loaded_models = getattr(backend, "get_loaded_models", lambda: [current] if current else [])()
    runtime = getattr(backend, "get_runtime_state", lambda: None)()
    state = runtime.get("state") if isinstance(runtime, dict) else ("loaded" if current else "unloaded")
    last_error = runtime.get("last_error") if isinstance(runtime, dict) else None
    return {
        "current": current,
        "available": available,
        "loaded_models": loaded_models,
        "state": state,
        "last_error": last_error,
        "runtime": runtime,
    }


@router.post("/backend/models/switch")
async def switch_backend_model(body: BackendModelSwitchRequest):
    """Switch the loaded model (optimized backend only)."""
    run_id = f"api-switch-{int(time.time() * 1000)}"
    # region agent log
    _debug_log(run_id, "H5", "openai_compatible.py:switch_backend_model:entry", "switch endpoint entry", {"model_key": body.model_key})
    # endregion
    from ..backends import get_backend

    backend = get_backend()
    if not hasattr(backend, "switch_model") or not hasattr(backend, "get_available_models"):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_supported",
                "message": "This backend does not support model switching.",
                "type": "invalid_request_error",
            },
        )
    available = backend.get_available_models()
    if body.model_key not in available:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model_key",
                "message": f"Unknown model_key '{body.model_key}'. Available: {available}",
                "type": "invalid_request_error",
            },
        )
    await backend.switch_model(body.model_key)
    loaded_models = getattr(backend, "get_loaded_models", lambda: [body.model_key])()
    runtime = getattr(backend, "get_runtime_state", lambda: None)()
    # region agent log
    _debug_log(
        run_id,
        "H4",
        "openai_compatible.py:switch_backend_model:success",
        "switch endpoint success",
        {
            "state": (runtime or {}).get("state", "loaded") if isinstance(runtime, dict) else "loaded",
            "current": body.model_key,
            "loaded_models": loaded_models,
            "last_error": (runtime or {}).get("last_error") if isinstance(runtime, dict) else None,
        },
    )
    # endregion
    return {
        "current": body.model_key,
        "loaded_models": loaded_models,
        "state": (runtime or {}).get("state", "loaded") if isinstance(runtime, dict) else "loaded",
    }


@router.post("/backend/models/unload")
async def unload_backend_model():
    """Unload current backend model from memory (when backend supports it)."""
    from ..backends import get_backend

    backend = get_backend()
    if not hasattr(backend, "unload_model"):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_supported",
                "message": "This backend does not support model unload.",
                "type": "invalid_request_error",
            },
        )
    await backend.unload_model()
    runtime = getattr(backend, "get_runtime_state", lambda: None)()
    return {
        "current": getattr(backend, "get_current_model_key", lambda: None)(),
        "loaded_models": getattr(backend, "get_loaded_models", lambda: [])(),
        "state": (runtime or {}).get("state", "unloaded") if isinstance(runtime, dict) else "unloaded",
    }


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    # Default voices (always available)
    default_voices = [
        VoiceInfo(id="Vivian", name="Vivian", language="English", description="Female voice"),
        VoiceInfo(id="Ryan", name="Ryan", language="English", description="Male voice"),
        VoiceInfo(id="Sophia", name="Sophia", language="English", description="Female voice"),
        VoiceInfo(id="Isabella", name="Isabella", language="English", description="Female voice"),
        VoiceInfo(id="Evan", name="Evan", language="English", description="Male voice"),
        VoiceInfo(id="Lily", name="Lily", language="English", description="Female voice"),
    ]
    
    # OpenAI-compatible voice aliases
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Sophia)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Isabella)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Evan)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Lily)"),
    ]
    
    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]
    
    try:
        backend = await get_tts_backend()
        
        # Get supported speakers from the backend
        speakers = backend.get_supported_voices()
        
        # Get supported languages
        languages = backend.get_supported_languages()
        
        # Build voice list from backend
        if speakers:
            voices = []
            for speaker in speakers:
                if backend.is_custom_voice(speaker):
                    description = f"Custom cloned voice: {speaker}"
                else:
                    description = f"Qwen3-TTS voice: {speaker}"
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=description,
                )
                voices.append(voice_info.model_dump())
        else:
            voices = [v.model_dump() for v in default_voices]
        
        # OpenAI aliases map to built-in speakers; skip them on Base models
        if backend.get_model_type() != "base":
            voices += [v.model_dump() for v in openai_voices]

        return {
            "voices": voices,
            "languages": languages if languages else default_languages,
        }
        
    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        # Return default voices if backend is not loaded
        return {
            "voices": [v.model_dump() for v in default_voices] + [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }


@router.get("/audio/voice-clone/capabilities")
async def get_voice_clone_capabilities():
    """
    Get voice cloning capabilities of the current backend.

    Returns whether voice cloning is supported and what modes are available.
    Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
    """
    try:
        backend = await get_tts_backend()

        supports_cloning = backend.supports_voice_cloning()
        model_type = backend.get_model_type() if hasattr(backend, 'get_model_type') else "unknown"

        return VoiceCloneCapabilities(
            supported=supports_cloning,
            model_type=model_type,
            icl_mode_available=supports_cloning,
            x_vector_mode_available=supports_cloning,
        )

    except Exception as e:
        logger.warning(f"Could not get voice clone capabilities: {e}")
        return VoiceCloneCapabilities(
            supported=False,
            model_type="unknown",
            icl_mode_available=False,
            x_vector_mode_available=False,
        )


@router.post("/audio/voice-clone")
async def create_voice_clone(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """
    Clone a voice from reference audio and generate speech.

    This endpoint requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
    Set TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable when starting the server.

    Two modes are available:
    - ICL mode (x_vector_only_mode=False): Requires ref_text transcript for best quality
    - X-Vector mode (x_vector_only_mode=True): No transcript needed, good quality
    """
    try:
        # vLLM-Omni proxy mode: forward non-streaming voice-clone to vLLM /v1/audio/speech
        if os.getenv("TTS_BACKEND", "").lower() in ("vllm_omni", "vllm-omni", "vllm") and os.getenv("VLLM_SERVE_URL"):
            if not request.ref_text and not request.x_vector_only_mode:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_ref_text",
                        "message": "ICL mode requires ref_text. Either provide ref_text or set x_vector_only_mode=True.",
                        "type": "invalid_request_error",
                    },
                )
            normalized_text = normalize_text(request.input, request.normalization_options)
            if not normalized_text.strip():
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid_input", "message": "Input text is empty after normalization", "type": "invalid_request_error"},
                )
            seed_used = (
                request.seed
                if request.seed is not None and request.seed >= 0
                else random.randint(0, 2**31 - 1)
            )
            payload = {
                "input": normalized_text,
                "voice": "vivian",
                "language": request.language or "Auto",
                "task_type": "Base",
                "ref_audio": _to_audio_data_url(request.ref_audio),
                "ref_text": request.ref_text or "",
                "x_vector_only_mode": bool(request.x_vector_only_mode),
                "stream": False,
                "response_format": request.response_format,
            }
            if seed_used >= 0:
                payload["seed"] = seed_used
            vllm_url = os.environ["VLLM_SERVE_URL"]
            audio_bytes = await proxy_vllm_speech(vllm_url, payload, timeout_s=300.0)
            content_type = get_content_type(request.response_format)
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}",
                    "Cache-Control": "no-cache",
                    "X-TTS-Seed": str(seed_used),
                },
            )

        backend = await get_tts_backend()

        # Check if voice cloning is supported
        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                               "Set TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable and restart the server.",
                    "type": "invalid_request_error",
                },
            )

        # Validate ICL mode requires ref_text
        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text (transcript of reference audio). "
                               "Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.ref_audio)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_audio",
                    "message": f"Failed to decode base64 audio: {e}",
                    "type": "invalid_request_error",
                },
            )

        # Load audio using soundfile
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)

            # Convert to mono if stereo
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)

            ref_audio = ref_audio.astype(np.float32)

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "audio_processing_error",
                    "message": f"Failed to process reference audio: {e}. "
                               "Ensure the audio is a valid WAV, MP3, or other supported format.",
                    "type": "invalid_request_error",
                },
            )

        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        # Resolve seed: -1 or None => random
        seed_used = (
            request.seed
            if request.seed is not None and request.seed >= 0
            else random.randint(0, 2**31 - 1)
        )

        clone_kwargs = {
            "text": normalized_text,
            "ref_audio": ref_audio,
            "ref_audio_sr": ref_sr,
            "ref_text": request.ref_text,
            "language": request.language or "Auto",
            "x_vector_only_mode": request.x_vector_only_mode,
        }
        if _method_accepts_kwarg(backend.generate_voice_clone, "speed"):
            clone_kwargs["speed"] = request.speed
        if _method_accepts_kwarg(backend.generate_voice_clone, "seed"):
            clone_kwargs["seed"] = seed_used
        if request.cache_key and _method_accepts_kwarg(backend.generate_voice_clone, "cache_key"):
            clone_kwargs["cache_key"] = request.cache_key

        # Generate voice clone without passing unsupported backend kwargs.
        audio, sample_rate = await backend.generate_voice_clone(**clone_kwargs)

        # Encode audio to requested format
        audio_bytes = encode_audio(audio, request.response_format, sample_rate)

        # Get content type
        content_type = get_content_type(request.response_format)

        # Return audio response with seed used in header
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}",
                "Cache-Control": "no-cache",
                "X-TTS-Seed": str(seed_used),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.post("/audio/voice-clone/stream")
async def create_voice_clone_stream(
    request: StreamingVoiceCloneRequest,
    client_request: Request,
):
    """
    Stream voice-cloned speech generation with real-time timing metrics.

    This endpoint streams audio chunks as they are generated, providing
    significantly lower latency for the first audio output.

    Returns WAV audio chunks with timing information in response headers.

    **Timing Headers:**
    - X-First-Chunk-Time: Time in seconds until first audio chunk
    - X-Total-Time: Total generation time in seconds
    - X-Audio-Duration: Duration of generated audio in seconds
    - X-RTF: Real-Time Factor (generation time / audio duration)
    - X-Chunk-Count: Number of audio chunks generated
    """
    try:
        # vLLM-Omni proxy mode: translate voice-clone streaming into /v1/audio/speech streaming.
        if os.getenv("TTS_BACKEND", "").lower() in ("vllm_omni", "vllm-omni", "vllm") and os.getenv("VLLM_SERVE_URL"):
            if request.speed != 1.0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_speed_for_streaming",
                        "message": "vLLM streaming does not support speed adjustment. Use speed=1.0.",
                        "type": "invalid_request_error",
                    },
                )
            vllm_url = os.environ["VLLM_SERVE_URL"]
            seed_used = (
                request.seed
                if request.seed is not None and request.seed >= 0
                else random.randint(0, 2**31 - 1)
            )
            payload = {
                "input": normalize_text(request.input, request.normalization_options),
                "voice": "vivian",
                "language": request.language or "Auto",
                "task_type": "Base",
                "ref_audio": _to_audio_data_url(request.ref_audio),
                "ref_text": request.ref_text or "",
                "x_vector_only_mode": bool(request.x_vector_only_mode),
                "stream": True,
                "response_format": "pcm",
            }
            if seed_used >= 0:
                payload["seed"] = seed_used
            return StreamingResponse(
                proxy_vllm_speech_stream_framed(
                    vllm_url, payload, timeout_s=300.0, seed_used=seed_used
                ),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": "attachment; filename=streaming_voice_clone.pcm",
                    "Cache-Control": "no-cache",
                    "X-Streaming": "true",
                },
            )

        backend = await get_tts_backend()

        # Check if voice cloning is supported
        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                               "Set TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable and restart the server.",
                    "type": "invalid_request_error",
                },
            )

        # Validate ICL mode requires ref_text
        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text (transcript of reference audio). "
                               "Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.ref_audio)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_audio",
                    "message": f"Failed to decode base64 audio: {e}",
                    "type": "invalid_request_error",
                },
            )

        # Load audio using soundfile
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)

            # Convert to mono if stereo
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)

            ref_audio = ref_audio.astype(np.float32)

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "audio_processing_error",
                    "message": f"Failed to process reference audio: {e}. "
                               "Ensure the audio is a valid WAV, MP3, or other supported format.",
                    "type": "invalid_request_error",
                },
            )

        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        seed_used = (
            request.seed
            if request.seed is not None and request.seed >= 0
            else random.randint(0, 2**31 - 1)
        )

        # Create async generator for streaming
        # Optimized backend uses generate_voice_clone_streaming; official uses stream_generate_voice_clone
        if hasattr(backend, "generate_voice_clone_streaming"):
            _stream_method = "generate_voice_clone_streaming"
            _stream_kwargs = {
                "text": normalized_text,
                "ref_audio": ref_audio,
                "ref_audio_sr": ref_sr,
                "ref_text": request.ref_text,
                "language": request.language or "Auto",
                "x_vector_only_mode": request.x_vector_only_mode,
            }
            if _method_accepts_kwarg(backend.generate_voice_clone_streaming, "cache_key"):
                _stream_kwargs["cache_key"] = request.cache_key
        else:
            _stream_method = "stream_generate_voice_clone"
            _stream_kwargs = {
                "text": normalized_text,
                "ref_audio": ref_audio,
                "ref_audio_sr": ref_sr,
                "ref_text": request.ref_text,
                "language": request.language or "Auto",
                "x_vector_only_mode": request.x_vector_only_mode,
                "speed": request.speed,
                "seed": seed_used,
                "emit_every_frames": request.emit_every_frames,
                "decode_window_frames": request.decode_window_frames,
            }

        async def audio_stream_generator():
            start_time = time.time()
            first_chunk_time = None
            chunk_count = 0
            total_samples = 0
            sample_rate = 24000  # Default, will be updated from chunks

            try:
                # Stream audio chunks from backend
                if _stream_method == "generate_voice_clone_streaming":
                    stream_iter = backend.generate_voice_clone_streaming(**_stream_kwargs)
                else:
                    stream_iter = backend.stream_generate_voice_clone(**_stream_kwargs)
                async for chunk, sr in stream_iter:
                    chunk_count += 1
                    sample_rate = sr
                    total_samples += len(chunk)

                    if first_chunk_time is None:
                        first_chunk_time = time.time()

                    # Encode chunk to raw PCM format (not WAV) to avoid header issues when concatenating
                    # The client will combine all PCM chunks and create a single WAV file
                    chunk_bytes = encode_audio(chunk, "pcm", sr)

                    # Yield chunk with timing metadata as JSON header
                    import json
                    timing = {
                        "chunk": chunk_count,
                        "first_chunk_time": first_chunk_time - start_time if first_chunk_time else None,
                    }
                    # Format: [4 bytes JSON length][JSON metadata][4 bytes audio length][audio bytes]
                    timing_json = json.dumps(timing).encode('utf-8')
                    import struct
                    yield struct.pack('<I', len(timing_json)) + timing_json + struct.pack('<I', len(chunk_bytes)) + chunk_bytes

                # Calculate final timing
                total_time = time.time() - start_time
                audio_duration = total_samples / sample_rate if sample_rate > 0 else 0
                rtf = total_time / audio_duration if audio_duration > 0 else 0

                # Send final timing info as last chunk (include seed_used for UI)
                final_timing = {
                    "done": True,
                    "first_chunk_time": first_chunk_time - start_time if first_chunk_time else None,
                    "total_time": total_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf,
                    "chunk_count": chunk_count,
                    "seed_used": seed_used,
                }
                timing_json = json.dumps(final_timing).encode('utf-8')
                import struct
                # Format: [4 bytes JSON length][JSON metadata][4 bytes audio length (0)][no audio]
                yield struct.pack('<I', len(timing_json)) + timing_json + struct.pack('<I', 0)

            except Exception as e:
                logger.error(f"Streaming generation error: {e}")
                import json
                import struct as struct_module
                error_data = {"error": str(e)}
                error_json = json.dumps(error_data).encode()
                # Format: [4 bytes JSON length][JSON metadata][4 bytes audio length (0)]
                yield struct_module.pack('<I', len(error_json)) + error_json + struct_module.pack('<I', 0)

        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=streaming_voice_clone.wav",
                "Cache-Control": "no-cache",
                "X-Streaming": "true",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming voice cloning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
