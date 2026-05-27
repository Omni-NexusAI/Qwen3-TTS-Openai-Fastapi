"""
Gradio Voice Studio for Qwen3-TTS.

This module defines a comprehensive Gradio-based UI that allows users to
create, manage, and export reusable voice profiles using the Qwen3-TTS
API.  The interface supports three primary workflows: preset voices
(CustomVoice), voice design (VoiceDesign), and voice cloning (Base).
Users can store profiles locally, preview and delete them, export
selected profiles to ZIP archives, and synthesize speech with saved
profiles via an interactive playground.

The entrypoint function ``build_app(base_url: str, library_dir: Path)``
returns a Gradio Blocks object configured with the desired API base URL
and profile storage directory.  See README or docstrings for usage.
"""

import argparse
import base64
import json
import mimetypes
import os
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import httpx


# -----------------------------------------------------------------------------
# Configuration and Defaults
# -----------------------------------------------------------------------------

# The default directory used to store voice profiles.  This can be overridden
# via the VOICE_LIBRARY_DIR environment variable or passed directly to
# ``build_app``.
DEFAULT_LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Base URL pointing to the running Qwen3-TTS API.  If not provided, this
# defaults to the local server on port 8880.  It may be overridden via the
# TTS_BASE_URL environment variable or passed to ``build_app``.
DEFAULT_TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880").rstrip("/")

# Default timeout for API requests in seconds.  Can be customized via the
# TTS_TIMEOUT_S environment variable or overridden at runtime.
DEFAULT_TIMEOUT_S = float(os.environ.get("TTS_TIMEOUT_S", "300"))

# Supported task types defined by the Qwen3-TTS API.  These values map to
# endpoint parameters used when creating and managing profiles.
SUPPORTED_TASK_TYPES = ["CustomVoice", "VoiceDesign", "Base"]

# A sample reference line used for voice design workflows.  This line will be
# synthesized as part of the profile creation process.
DEFAULT_REFERENCE_LINE = (
    "Hi! This is a reference clip for my custom voice. "
    "I will reuse this voice for future speech synthesis."
)

# Fallback voices list used when the server does not provide a voices
# endpoint or fails to return names.
FALLBACK_VOICES = ["Vivian", "Ryan", "Serena", "Dylan", "Eric", "Aiden"]


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "35990f",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
        }
        httpx.post(
            "http://127.0.0.1:7852/ingest/d65a6209-8a13-4b06-b1f6-8a0a5aeed5c8",
            headers={"Content-Type": "application/json", "X-Debug-Session-Id": "35990f"},
            json=payload,
            timeout=1.0,
        )
    except Exception:
        pass


@dataclass
class VoiceProfile:
    """Representation of a saved voice profile.

    Attributes:
        profile_id: Unique identifier for the profile.
        name: Human-friendly name assigned by the user.
        task_type: Type of profile (CustomVoice, VoiceDesign, Base).
        created_at: ISO8601 timestamp indicating when the profile was created.
        language: Language code or Auto for automatic detection.
        voice: Name of the voice (for CustomVoice or design clones).
        instructions: Style instructions (for CustomVoice or voice design).
        ref_text: Transcript of the reference audio (for Base profiles).
        x_vector_only_mode: Whether the clone is created without a transcript.
        ref_audio_filename: Filename of the reference audio stored in the profile directory.
        origin: Human-readable descriptor of how the profile was created.
    """

    profile_id: str
    name: str
    task_type: str
    created_at: str
    language: str = "Auto"
    voice: str = "Vivian"
    instructions: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = False
    ref_audio_filename: str = ""
    origin: str = ""


def ensure_dirs(library_dir: Path) -> Dict[str, Path]:
    """Ensure that the profiles and exports directories exist.

    Returns a dictionary mapping directory names to Path objects.
    """
    profiles_dir = library_dir / "profiles"
    exports_dir = library_dir / "exports"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    return {"profiles": profiles_dir, "exports": exports_dir}


def now_iso() -> str:
    """Return the current UTC time as an ISO8601-formatted string."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_profile_id() -> str:
    """Generate a short, unique profile identifier."""
    return uuid.uuid4().hex[:12]


def profile_dir(library_dir: Path, profile_id: str) -> Path:
    """Return the path to a given profile's directory."""
    return ensure_dirs(library_dir)["profiles"] / profile_id


def meta_path(library_dir: Path, profile_id: str) -> Path:
    """Return the path to a profile's metadata file."""
    return profile_dir(library_dir, profile_id) / "meta.json"


def load_profile(library_dir: Path, profile_id: str) -> VoiceProfile:
    """Load a profile from disk into a VoiceProfile instance."""
    p = meta_path(library_dir, profile_id)
    data = json.loads(p.read_text(encoding="utf-8"))
    return VoiceProfile(**data)


def save_profile(library_dir: Path, vp: VoiceProfile) -> None:
    """Persist a VoiceProfile to disk by writing its metadata file."""
    d = profile_dir(library_dir, vp.profile_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(vp.__dict__, indent=2), encoding="utf-8")


def delete_profile(library_dir: Path, profile_id: str) -> None:
    """Remove a profile directory and all its contents."""
    d = profile_dir(library_dir, profile_id)
    if d.exists():
        shutil.rmtree(d)


def list_profiles(library_dir: Path) -> List[VoiceProfile]:
    """Return all profiles stored in the library sorted by creation time."""
    dirs = ensure_dirs(library_dir)["profiles"]
    out: List[VoiceProfile] = []
    for child in sorted(dirs.iterdir(), key=lambda p: p.name):
        if child.is_dir():
            mp = child / "meta.json"
            if mp.exists():
                try:
                    data = json.loads(mp.read_text(encoding="utf-8"))
                    out.append(VoiceProfile(**data))
                except Exception:
                    pass  # skip corrupted entries
    # Sort newest first
    out.sort(key=lambda x: x.created_at, reverse=True)
    return out


def profiles_table_rows(profiles: List[VoiceProfile]) -> List[List[Any]]:
    """Convert profiles list into table rows for the library tab."""
    rows = []
    for p in profiles:
        rows.append([
            p.profile_id,
            p.name,
            p.task_type,
            p.origin,
            p.language,
            p.voice,
            (p.instructions[:60] + "…") if len(p.instructions) > 60 else p.instructions,
            "yes" if bool(p.ref_audio_filename) else "no",
            p.created_at,
        ])
    return rows


def normalize_base_url(base_url: str) -> str:
    """Ensure the base URL does not end with a trailing slash."""
    return base_url.rstrip("/")


def data_uri_from_file(file_path: Path) -> str:
    """Encode a file's contents as a data URI with guessed MIME type."""
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        mime = "audio/wav"
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def base64_from_file(file_path: Path) -> str:
    """Encode a file's contents as base64 string (no data URI prefix)."""
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def write_bytes_to_temp_audio(content: bytes, ext: str) -> str:
    """Write raw audio bytes to a temporary file and return its path."""
    ext = ext.lstrip(".")
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    os.close(fd)
    Path(path).write_bytes(content)
    return path


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Convert raw PCM bytes to WAV format with proper header."""
    import struct
    import io
    
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = len(pcm_bytes)
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')
    
    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # Data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(pcm_bytes)
    
    return buffer.getvalue()


def request_tts(base_url: str, payload: Dict[str, Any], timeout_s: float) -> Tuple[bytes, str]:
    """Call the /v1/audio/speech endpoint and return audio bytes and extension."""
    url = normalize_base_url(base_url) + "/v1/audio/speech"
    response_format = payload.get("response_format") or "wav"
    payload["response_format"] = response_format
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
    ext = response_format.lower()
    if ext == "pcm":
        ext = "raw"
    return r.content, ext


def request_tts_voice_clone(
    base_url: str, payload: Dict[str, Any], timeout_s: float
) -> Tuple[bytes, str, Dict[str, Any]]:
    """Call the /v1/audio/voice-clone endpoint (non-streaming). Returns (audio_bytes, extension, headers_dict)."""
    url = normalize_base_url(base_url) + "/v1/audio/voice-clone"
    response_format = payload.get("response_format") or "wav"
    payload["response_format"] = response_format
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
    ext = response_format.lower()
    if ext == "pcm":
        ext = "raw"
    headers = dict(r.headers) if r.headers else {}
    return r.content, ext, headers


def request_tts_streaming(
    base_url: str, 
    payload: Dict[str, Any], 
    timeout_s: float
) -> Tuple[bytes, str, Dict[str, Any]]:
    """
    Call the /v1/audio/voice-clone/stream endpoint and return audio bytes, extension, and timing info.
    
    Returns:
        Tuple of (audio_bytes, extension, timing_info)
    """
    import struct
    import json as json_module
    
    url = normalize_base_url(base_url) + "/v1/audio/voice-clone/stream"
    
    with httpx.Client(timeout=timeout_s) as client:
        with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            chunks = []
            timing_info = {}
            buffer = b""
            
            for chunk in response.iter_bytes():
                buffer += chunk
                
                # Process complete chunks
                # Format: [4 bytes JSON length][JSON metadata][4 bytes audio length][audio bytes]
                while len(buffer) >= 4:
                    json_len = struct.unpack('<I', buffer[:4])[0]
                    if len(buffer) < 4 + json_len + 4:
                        break  # Need more data for JSON + audio length
                    
                    json_bytes = buffer[4:4 + json_len]
                    metadata = json_module.loads(json_bytes.decode('utf-8'))
                    audio_len = struct.unpack('<I', buffer[4 + json_len:4 + json_len + 4])[0]
                    
                    if len(buffer) < 4 + json_len + 4 + audio_len:
                        break  # Need more data for audio bytes
                    
                    audio_bytes = buffer[4 + json_len + 4:4 + json_len + 4 + audio_len]
                    buffer = buffer[4 + json_len + 4 + audio_len:]  # Keep remaining data
                    
                    if metadata.get("error"):
                        raise RuntimeError(metadata["error"])
                    
                    if metadata.get("done"):
                        timing_info = {
                            "first_chunk_time": metadata.get("first_chunk_time"),
                            "total_time": metadata.get("total_time"),
                            "audio_duration": metadata.get("audio_duration"),
                            "rtf": metadata.get("rtf"),
                            "chunk_count": metadata.get("chunk_count"),
                            "seed_used": metadata.get("seed_used"),
                        }
                    elif len(audio_bytes) > 0:
                        chunks.append(audio_bytes)
            
            # Combine all audio chunks (raw PCM data)
            total_length = sum(len(c) for c in chunks)
            combined_pcm = b"".join(chunks)
            
            # Convert raw PCM to WAV format with proper header
            # PCM is 16-bit signed integers at 24000 Hz
            wav_bytes = pcm_to_wav(combined_pcm, sample_rate=24000)
            
            return wav_bytes, "wav", timing_info


def try_fetch_voices(base_url: str, timeout_s: float) -> List[str]:
    """Attempt to fetch available voices from the /v1/voices endpoint."""
    url = normalize_base_url(base_url) + "/v1/voices"
    try:
        with httpx.Client(timeout=min(timeout_s, 20.0)) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
        if isinstance(data, dict) and isinstance(data.get("voices"), list):
            return [str(x) for x in data["voices"]]
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            names: List[str] = []
            for item in data["data"]:
                if isinstance(item, dict) and "name" in item:
                    names.append(str(item["name"]))
            if names:
                return names
    except Exception:
        pass
    return FALLBACK_VOICES


def export_profiles_zip(library_dir: Path, profile_ids: Optional[List[str]] = None) -> str:
    """Create a ZIP archive of selected profiles (or all) and return its path."""
    ensure_dirs(library_dir)
    profiles = list_profiles(library_dir)
    if profile_ids:
        profiles = [p for p in profiles if p.profile_id in set(profile_ids)]
    exports_dir = ensure_dirs(library_dir)["exports"]
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_path = exports_dir / f"voices_export_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        manifest: List[Dict[str, Any]] = []
        for p in profiles:
            d = profile_dir(library_dir, p.profile_id)
            z.write(d / "meta.json", arcname=f"{p.profile_id}/meta.json")
            if p.ref_audio_filename:
                ref_path = d / p.ref_audio_filename
                if ref_path.exists():
                    z.write(ref_path, arcname=f"{p.profile_id}/{p.ref_audio_filename}")
            manifest.append(p.__dict__)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    return str(zip_path)


def import_profiles_zip(library_dir: Path, zip_path: Path) -> Dict[str, Any]:
    """Import profiles from an export ZIP and return a summary."""
    if not zip_path.exists():
        raise ValueError(f"ZIP not found: {zip_path}")

    ensure_dirs(library_dir)
    imported = 0
    skipped = 0
    errors: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        profile_roots = sorted({n.split("/", 1)[0] for n in names if "/" in n and n != "MANIFEST.json"})
        for root in profile_roots:
            meta_name = f"{root}/meta.json"
            if meta_name not in names:
                skipped += 1
                errors.append(f"{root}: missing meta.json")
                continue
            try:
                meta_raw = z.read(meta_name).decode("utf-8")
                meta = json.loads(meta_raw)
            except Exception as exc:
                skipped += 1
                errors.append(f"{root}: invalid meta.json ({exc})")
                continue

            try:
                vp = VoiceProfile(**meta)
            except Exception as exc:
                skipped += 1
                errors.append(f"{root}: invalid profile fields ({exc})")
                continue

            # Always allocate a fresh profile_id to avoid collisions/overwrites.
            new_id = safe_profile_id()
            while profile_dir(library_dir, new_id).exists():
                new_id = safe_profile_id()
            old_ref_filename = vp.ref_audio_filename
            vp.profile_id = new_id

            dest_dir = profile_dir(library_dir, vp.profile_id)
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy referenced audio if present in archive.
            if old_ref_filename:
                ref_member = f"{root}/{old_ref_filename}"
                if ref_member in names:
                    (dest_dir / old_ref_filename).write_bytes(z.read(ref_member))
                else:
                    vp.ref_audio_filename = ""
                    errors.append(f"{root}: referenced audio '{old_ref_filename}' missing in ZIP")

            save_profile(library_dir, vp)
            imported += 1

    return {"imported": imported, "skipped": skipped, "errors": errors}


# -----------------------------------------------------------------------------
# Gradio UI Construction
# -----------------------------------------------------------------------------

# Custom CSS to style the interface with a dark theme that complements the
# existing web UI of the Qwen3-TTS server.
CSS = """
:root {
  /* Light-skew palette to match Gradio Soft defaults */
  --bg0: #f7f8fc;
  --bg1: #ffffff;
  --card: rgba(255,255,255,0.95);
  --card2: rgba(255,255,255,0.98);
  --border: rgba(15,23,42,0.12);
  --text: #0f172a;
  --muted: rgba(15,23,42,0.72);
}
body, .gradio-container { background: radial-gradient(1200px 800px at 10% 10%, var(--bg1), var(--bg0)) !important; }
body, .gradio-container { color: var(--text) !important; color-scheme: light; }
#header {
  padding: 18px 18px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(59,130,246,0.06));
  border-radius: 18px;
}
.card {
  border: 1px solid var(--border);
  background: var(--card);
  border-radius: 18px;
}
.small { color: var(--muted) !important; font-size: 0.95em; }

/* Keep menus readable without forcing a dark theme */
.gradio-container label, .gradio-container .label, .gradio-container .wrap label { color: var(--muted) !important; }
.gradio-container .prose, .gradio-container .markdown, .gradio-container .md { color: var(--text) !important; }

/* Subtle surfaces: do not override Gradio component colors heavily */
.gradio-container .block, .gradio-container .panel, .gradio-container .gr-box, .gradio-container .wrap {
  border-color: var(--border) !important;
}
"""

TABLE_HEADERS = [
    "id",
    "name",
    "task_type",
    "origin",
    "language",
    "voice",
    "instructions",
    "has_ref_audio",
    "created_at",
]


def build_app(initial_base_url: str, initial_library_dir: Path) -> gr.Blocks:
    """Construct the Gradio Blocks interface for the Voice Studio.

    Args:
        initial_base_url: Base URL of the Qwen3-TTS API to use for requests.
        initial_library_dir: Directory where profiles will be stored.

    Returns:
        A gradio.Blocks object ready to be mounted on a FastAPI server or
        launched standalone via ``.launch()``.
    """
    ensure_dirs(initial_library_dir)
    with gr.Blocks(title="Qwen3 Voice Studio", css=CSS, theme=gr.themes.Soft()) as demo:
        # Shared state variables
        state_base_url = gr.State(initial_base_url)
        state_library_dir = gr.State(str(initial_library_dir))
        state_voices = gr.State([])

        # Header section
        gr.HTML(
            """
            <div id="header">
              <div style="font-size: 1.35rem; font-weight: 700;">Qwen3 Voice Studio</div>
              <div class="small">
                Create & save reusable voice profiles (preset, designed, or cloned) and export them for inference.
              </div>
            </div>
            """
        )

        # Settings accordion
        with gr.Accordion("Settings", open=False):
            with gr.Row():
                base_url_in = gr.Textbox(
                    label="TTS Server Base URL",
                    value=initial_base_url,
                    placeholder="http://localhost:8880",
                )
                library_dir_in = gr.Textbox(
                    label="Voice Library Dir",
                    value=str(initial_library_dir),
                    placeholder="./voice_library",
                )
                timeout_in = gr.Number(
                    label="Request timeout (seconds)",
                    value=DEFAULT_TIMEOUT_S,
                    precision=0,
                )
            with gr.Row():
                refresh_voices_btn = gr.Button("Refresh voices from server", variant="primary")
                voices_status = gr.Markdown("", elem_classes=["small"])
            with gr.Row():
                backend_status_md = gr.Markdown("", elem_classes=["small"])
            # Backend model (optimized backend only): show when GET /v1/backend/models returns list
            backend_model_column = gr.Column(visible=False)
            with backend_model_column:
                gr.Markdown("**Backend model controls**")
                backend_model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=None,
                )
                with gr.Row():
                    switch_model_btn = gr.Button("Load selected model", variant="secondary")
                    unload_model_btn = gr.Button("Unload model", variant="stop")
                    refresh_model_status_btn = gr.Button("Refresh model status", variant="secondary")
                backend_models_status_md = gr.Markdown("", elem_classes=["small"])

        # Global log output
        with gr.Row():
            global_log = gr.Markdown("", elem_classes=["small"])

        # Tabs for create, library, and playground
        with gr.Tabs():
            # Create tab and nested subtabs
            with gr.Tab("Create"):
                with gr.Tabs():
                    # Preset (CustomVoice)
                    with gr.Tab("Preset Voice (CustomVoice)"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                preset_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Vivian - Friendly NZ Support'")
                                preset_voice = gr.Dropdown(label="Voice", choices=FALLBACK_VOICES, value=FALLBACK_VOICES[0])
                                preset_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                preset_instructions = gr.Textbox(
                                    label="Style instructions (optional)",
                                    placeholder="e.g. Calm, confident, slightly upbeat, clear articulation.",
                                    lines=3,
                                )
                                preset_test_text = gr.Textbox(
                                    label="Test text",
                                    value="Hello! This is my saved preset voice profile.",
                                    lines=3,
                                )
                                preset_generate_btn = gr.Button("Generate", variant="primary")
                                preset_save_btn = gr.Button("Save profile", variant="secondary")
                            with gr.Column(scale=1, min_width=320):
                                preset_audio = gr.Audio(label="Output audio (trimmable)", type="filepath", editable=True)
                                preset_download = gr.File(label="Download audio")

                    # Voice design (VoiceDesign)
                    with gr.Tab("Voice Design (generate reference clip → save)"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                design_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Warm storyteller (designed)'")
                                design_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                design_instructions = gr.Textbox(
                                    label="Voice description / instructions",
                                    placeholder="e.g. A warm, friendly female voice, mid-30s, clear diction, gentle energy.",
                                    lines=4,
                                )
                                design_ref_line = gr.Textbox(
                                    label="Reference line to synthesize (this gets saved as the transcript)",
                                    value=DEFAULT_REFERENCE_LINE,
                                    lines=3,
                                )
                                design_generate_btn = gr.Button("Generate reference clip", variant="primary")
                                design_save_as_clone_btn = gr.Button("Save as reusable clone profile", variant="secondary")
                            with gr.Column(scale=1, min_width=320):
                                design_audio = gr.Audio(label="Reference audio (output, trimmable)", type="filepath", editable=True)
                                design_download = gr.File(label="Download reference audio")

                    # Voice clone (Base)
                    with gr.Tab("Voice Clone (Base)"):
                        with gr.Row():
                            with gr.Column(scale=1, min_width=320):
                                clone_name = gr.Textbox(label="Profile name", placeholder="e.g. 'Facu - Mic Clone v1'")
                                clone_language = gr.Dropdown(
                                    label="Language",
                                    choices=[
                                        "Auto",
                                        "English",
                                        "Spanish",
                                        "Chinese",
                                        "Japanese",
                                        "Korean",
                                        "German",
                                        "French",
                                        "Russian",
                                        "Portuguese",
                                        "Italian",
                                    ],
                                    value="Auto",
                                )
                                clone_ref_audio = gr.Audio(
                                    label="Reference audio (upload or record)",
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                )
                                clone_xvec_only = gr.Checkbox(
                                    label="x_vector_only_mode (no transcript needed, usually lower quality)",
                                    value=False,
                                )
                                clone_ref_text = gr.Textbox(
                                    label="Reference transcript (recommended)",
                                    placeholder="Paste the transcript of the reference audio (or leave blank if x_vector_only_mode).",
                                    lines=3,
                                )
                                clone_test_text = gr.Textbox(
                                    label="Test text (what you want to synthesize)",
                                    value="Hello! This is a voice clone test.",
                                    lines=3,
                                )
                                clone_generate_btn = gr.Button("Generate", variant="primary")
                                clone_save_btn = gr.Button("Save clone profile", variant="secondary")
                            with gr.Column(scale=1, min_width=320):
                                clone_audio = gr.Audio(label="Output audio (trimmable)", type="filepath", editable=True)
                                clone_download = gr.File(label="Download audio")

            # Library tab
            with gr.Tab("Library"):
                with gr.Row():
                    with gr.Column(scale=2, min_width=520):
                        library_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=["str"] * len(TABLE_HEADERS),
                            label="Saved profiles",
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Row():
                            library_refresh_btn = gr.Button("Refresh list", variant="primary")
                            export_selected_btn = gr.Button("Export selected → ZIP", variant="secondary")
                            export_all_btn = gr.Button("Export ALL → ZIP", variant="secondary")
                        export_file = gr.File(label="Export download")
                        with gr.Row():
                            import_zip_file = gr.File(label="Import profile ZIP", file_types=[".zip"], type="filepath")
                            import_zip_btn = gr.Button("Import ZIP", variant="secondary")
                    with gr.Column(scale=1, min_width=340):
                        selected_id = gr.Textbox(label="Selected profile id", placeholder="Click a row to copy id here")
                        load_selected_btn = gr.Button("Load selected", variant="primary")
                        delete_selected_btn = gr.Button("Delete selected", variant="stop")
                        profile_details = gr.JSON(label="Profile details")
                        ref_preview = gr.Audio(label="Reference audio preview (if available)", type="filepath")

            # Playground tab
            with gr.Tab("Playground"):
                gr.Markdown(
                    "**Baseline comparison:** To compare 0.6B vs 1.7B, switch the model in **Settings → Backend model controls**, then run the same text here and check the timing table below.",
                    elem_classes=["small"],
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=360):
                        play_profile_id = gr.Dropdown(label="Pick a saved profile", choices=[], value=None)
                        play_text = gr.Textbox(label="Text to synthesize", value="Hello from the playground!", lines=4)
                        play_response_format = gr.Dropdown(
                            label="Audio format",
                            choices=["wav", "mp3", "flac", "aac", "opus", "pcm"],
                            value="wav",
                        )
                        play_speed = gr.Slider(label="Speed", minimum=0.25, maximum=4.0, value=1.0, step=0.05)
                        play_seed = gr.Number(
                            label="Seed (-1 = random)",
                            value=-1,
                            precision=0,
                            minimum=-1,
                            maximum=2147483647,
                        )
                        play_generate_btn = gr.Button("🎙️ Generate (Streaming)", variant="primary")
                    with gr.Column(scale=1, min_width=360):
                        play_audio = gr.Audio(label="Output audio (trimmable)", type="filepath", editable=True)
                        play_download = gr.File(label="Download audio")
                        # Timing display
                        play_timing = gr.Markdown(
                            value="",
                        )

        # ------------------------------------------------------------------
        # Callback implementations
        # ------------------------------------------------------------------

        def on_refresh_voices(base_url: str, timeout_s: float):
            voices = try_fetch_voices(base_url, float(timeout_s))
            return (
                gr.State(base_url),
                gr.State(voices),
                gr.Dropdown(choices=voices, value=voices[0] if voices else None),
                f"✅ Loaded {len(voices)} voices from server (or fallback list).",
            )

        def fetch_backend_status(base_url: str):
            """Fetch /health for quick model verification."""
            try:
                url = f"{base_url.rstrip('/')}/health"
                r = httpx.get(url, timeout=10.0)
                r.raise_for_status()
                data = r.json()
                backend = data.get("backend") or {}
                name = backend.get("name", "unknown")
                model_id = backend.get("model_id", "unknown")
                current_key = backend.get("current_model_key")
                loaded_models = backend.get("loaded_models") or []
                runtime = backend.get("runtime") or {}
                state = runtime.get("state")
                parts = [f"**Loaded backend:** `{name}`", f"**Model:** `{model_id}`"]
                if current_key:
                    parts.append(f"**Current key:** `{current_key}`")
                if loaded_models:
                    parts.append(f"**In memory:** `{', '.join(loaded_models)}`")
                if state:
                    parts.append(f"**State:** `{state}`")
                return " | ".join(parts)
            except Exception as e:
                return f"⚠️ Could not fetch `/health`: `{e}`"

        def load_backend_models(base_url: str):
            """Fetch backend model list; show column and dropdown; display loading/error state; enable/disable Generate buttons."""
            try:
                run_id = f"ui-models-{int(datetime.utcnow().timestamp() * 1000)}"
                # region agent log
                _debug_log(run_id, "H1", "gradio_voice_studio.py:load_backend_models:entry", "load_backend_models entry", {"base_url": base_url})
                # endregion
                url = f"{base_url.rstrip('/')}/v1/backend/models"
                r = httpx.get(url, timeout=10.0)
                if r.status_code != 200:
                    # region agent log
                    _debug_log(run_id, "H2", "gradio_voice_studio.py:load_backend_models:non200", "backend models non-200", {"status_code": r.status_code, "url": url})
                    # endregion
                    return (
                        gr.update(visible=False),
                        gr.update(choices=[], value=None),
                        "Model controls are unavailable.",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )
                data = r.json()
                available = [m for m in (data.get("available") or []) if "Base" in m]
                if not available:
                    # region agent log
                    _debug_log(run_id, "H3", "gradio_voice_studio.py:load_backend_models:no_base", "no Base models returned", {"available_raw": data.get("available"), "state": data.get("state")})
                    # endregion
                    return (
                        gr.update(visible=False),
                        gr.update(choices=[], value=None),
                        "No Base model controls available for this backend.",
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                    )
                current = data.get("current")
                loaded = data.get("loaded_models") or []
                state = data.get("state", "unknown")
                err = data.get("last_error")
                runtime = data.get("runtime") or {}
                last_load_s = runtime.get("last_load_elapsed_s")

                # State line: loading / ready / error
                if state == "loading":
                    status_parts = ["**State:** ⏳ Loading… (wait for completion or check logs)"]
                elif state == "loaded":
                    status_parts = ["**State:** ✅ Loaded — generation enabled"]
                elif state == "error":
                    status_parts = ["**State:** ❌ Error — load a model before generating"]
                else:
                    status_parts = [f"**State:** `{state}` — load a model to enable generation"]

                status_parts.append(f"**Selected (dropdown):** `{current or 'none'}`")
                status_parts.append(f"**In memory:** `{', '.join(loaded) if loaded else 'none'}`")
                if last_load_s is not None:
                    status_parts.append(f"**Last load time:** {last_load_s}s")
                if err:
                    status_parts.append(f"**Last error:** `{err}`")

                model_ready = state == "loaded"
                # region agent log
                _debug_log(
                    run_id,
                    "H4",
                    "gradio_voice_studio.py:load_backend_models:success",
                    "backend models loaded",
                    {"state": state, "current": current, "loaded_models": loaded, "model_ready": model_ready, "last_error": err},
                )
                # endregion
                return (
                    gr.update(visible=True),
                    gr.update(choices=available, value=current or available[0]),
                    "\n\n".join(status_parts),
                    gr.update(interactive=model_ready),
                    gr.update(interactive=model_ready),
                )
            except Exception as e:
                return (
                    gr.update(visible=False),
                    gr.update(choices=[], value=None),
                    f"⚠️ Failed to load model controls: {e}",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )

        def do_switch_backend_model(base_url: str, model_key: Optional[str]):
            """POST switch model; return updated dropdown, status text, and Generate button interactive state."""
            if not model_key:
                return gr.update(), "Select a model first.", gr.update(), gr.update()
            try:
                run_id = f"ui-switch-{int(datetime.utcnow().timestamp() * 1000)}"
                # region agent log
                _debug_log(run_id, "H5", "gradio_voice_studio.py:do_switch_backend_model:entry", "switch requested", {"base_url": base_url, "model_key": model_key})
                # endregion
                url = f"{base_url.rstrip('/')}/v1/backend/models/switch"
                r = httpx.post(url, json={"model_key": model_key}, timeout=600.0)
                # region agent log
                _debug_log(run_id, "H5", "gradio_voice_studio.py:do_switch_backend_model:switch_response", "switch response received", {"status_code": r.status_code})
                # endregion
                if r.status_code != 200:
                    err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    msg = err.get("detail", {}).get("message", "Switch failed") if isinstance(err.get("detail"), dict) else str(err.get("detail", "Switch failed"))
                    raise gr.Error(msg)
                r2 = httpx.get(f"{base_url.rstrip('/')}/v1/backend/models", timeout=10.0)
                if r2.status_code != 200:
                    return gr.update(value=model_key), f"✅ Loaded: `{model_key}`", gr.update(interactive=True), gr.update(interactive=True)
                data = r2.json()
                available = [m for m in (data.get("available", []) or []) if "Base" in m]
                loaded = data.get("loaded_models") or []
                state = data.get("state", "loaded")
                model_ready = state == "loaded"
                msg = f"✅ Loaded: `{data.get('current', model_key)}` | In memory: `{', '.join(loaded) if loaded else 'none'}`"
                # region agent log
                _debug_log(run_id, "H4", "gradio_voice_studio.py:do_switch_backend_model:post_status", "post-switch models status", {"state": state, "current": data.get("current", model_key), "loaded_models": loaded})
                # endregion
                return (
                    gr.update(choices=available, value=data.get("current", model_key)),
                    msg,
                    gr.update(interactive=model_ready),
                    gr.update(interactive=model_ready),
                )
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(str(e))

        def ensure_backend_model(base_url: str, model_key: Optional[str]) -> Optional[str]:
            """Validate that the selected model is already loaded before generation."""
            if not model_key:
                raise gr.Error("Select a model, then click Load selected model before generating.")

            status_url = f"{base_url.rstrip('/')}/v1/backend/models"
            try:
                status = httpx.get(status_url, timeout=10.0)
                if status.status_code != 200:
                    raise gr.Error(f"Could not verify loaded model: HTTP {status.status_code}")
                data = status.json()
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(f"Could not verify loaded model: {e}")

            loaded = data.get("loaded_models") or []
            current = data.get("current")
            state = data.get("state")
            if current == model_key and state == "loaded" and model_key in loaded:
                return model_key

            if state == "unloaded" or not loaded:
                raise gr.Error(f"`{model_key}` is selected but no model is loaded. Click Load selected model first.")
            raise gr.Error(
                f"`{model_key}` is selected but `{current or 'none'}` is loaded. "
                "Click Load selected model before generating."
            )

        def on_library_table_select(table: Any, evt: gr.SelectData):
            """Copy clicked Library id cell into the Selected profile id textbox."""
            try:
                row_idx, col_idx = evt.index
                id_col = TABLE_HEADERS.index("id")
                if col_idx != id_col:
                    return gr.update()
                if evt.value:
                    return str(evt.value)
                if hasattr(table, "iloc"):
                    value = table.iloc[row_idx, id_col]
                else:
                    value = table[row_idx][id_col]
                return str(value) if value is not None else gr.update()
            except Exception:
                return gr.update()

        def do_unload_backend_model(base_url: str):
            """POST unload current backend model; return status and disable Generate buttons."""
            try:
                url = f"{base_url.rstrip('/')}/v1/backend/models/unload"
                r = httpx.post(url, timeout=60.0)
                if r.status_code != 200:
                    err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                    msg = err.get("detail", {}).get("message", "Unload failed") if isinstance(err.get("detail"), dict) else str(err.get("detail", "Unload failed"))
                    raise gr.Error(msg)
                return "✅ Model unloaded. In memory: `none` — load a model to enable generation.", gr.update(interactive=False), gr.update(interactive=False)
            except gr.Error:
                raise
            except Exception as e:
                raise gr.Error(str(e))

        def on_generate_preset(base_url: str, timeout_s: float, voice: str, language: str, instructions: str, text: str):
            payload = {
                "input": text,
                "voice": voice,
                "language": language,
                "task_type": "CustomVoice",
                "instructions": instructions or "",
                "response_format": "wav",
            }
            audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
            out_path = write_bytes_to_temp_audio(audio_bytes, ext)
            return out_path, out_path, "✅ Generated audio."

        def on_save_preset(library_dir_str: str, name: str, voice: str, language: str, instructions: str):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            vp = VoiceProfile(
                profile_id=safe_profile_id(),
                name=name.strip(),
                task_type="CustomVoice",
                origin="Preset(CustomVoice)",
                created_at=now_iso(),
                language=language,
                voice=voice,
                instructions=instructions or "",
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved preset profile: {vp.profile_id}"

        def on_generate_design_ref(base_url: str, timeout_s: float, language: str, instructions: str, ref_line: str):
            if not instructions.strip():
                raise gr.Error("Voice design instructions are required.")
            payload = {
                "input": ref_line,
                "voice": "Vivian",
                "language": language,
                "task_type": "VoiceDesign",
                "instructions": instructions.strip(),
                "response_format": "wav",
            }
            audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
            out_path = write_bytes_to_temp_audio(audio_bytes, ext)
            return out_path, out_path, "✅ Generated reference clip."

        def on_save_design_as_clone(
            library_dir_str: str,
            name: str,
            language: str,
            instructions: str,
            ref_line: str,
            ref_audio_path: str,
        ):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Generate a reference clip first.")
            pid = safe_profile_id()
            dest_dir = profile_dir(Path(library_dir_str), pid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            ref_ext = Path(ref_audio_path).suffix or ".wav"
            ref_filename = f"ref_audio{ref_ext}"
            shutil.copy2(ref_audio_path, dest_dir / ref_filename)
            vp = VoiceProfile(
                profile_id=pid,
                name=name.strip(),
                task_type="Base",
                origin="VoiceDesign->Base",
                created_at=now_iso(),
                language=language,
                voice="Vivian",
                instructions=instructions.strip(),
                ref_text=ref_line,
                x_vector_only_mode=False,
                ref_audio_filename=ref_filename,
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved designed voice as reusable clone profile: {pid}"

        def on_generate_clone(
            base_url: str,
            timeout_s: float,
            selected_model: Optional[str],
            language: str,
            ref_audio_path: str,
            ref_text: str,
            xvec_only: bool,
            text: str,
        ):
            active_model = ensure_backend_model(base_url, selected_model)
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Reference audio is required.")
            if (not xvec_only) and (not ref_text.strip()):
                raise gr.Error("Reference transcript is required unless x_vector_only_mode is enabled.")
            ref_b64 = base64_from_file(Path(ref_audio_path))
            payload = {
                "input": text,
                "voice": "Vivian",
                "language": language,
                "task_type": "Base",
                "ref_audio": ref_b64,
                "ref_text": ref_text.strip(),
                "x_vector_only_mode": bool(xvec_only),
                "response_format": "wav",
            }
            # Use streaming endpoint for better timing info
            try:
                audio_bytes, ext, timing_info = request_tts_streaming(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                
                # Format timing info as readable markdown
                first_chunk = timing_info.get('first_chunk_time')
                total_time = timing_info.get('total_time')
                audio_duration = timing_info.get('audio_duration')
                rtf = timing_info.get('rtf')
                chunk_count = timing_info.get('chunk_count', 0)
                
                timing_md = f"""### ⏱️ Generation Timing
| Metric | Value |
|--------|-------|
| **First chunk** | {first_chunk:.2f}s |
| **Total time** | {total_time:.2f}s |
| **Audio duration** | {audio_duration:.2f}s |
| **RTF** | {rtf:.2f}x |
| **Chunks** | {chunk_count} |
| **Model** | {active_model or "current"} |
"""
                return out_path, out_path, timing_md
            except Exception as e:
                # Fallback to non-streaming
                audio_bytes, ext, _headers = request_tts_voice_clone(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                return out_path, out_path, f"⚠️ Used non-streaming fallback: {e}"

        def on_save_clone_profile(
            library_dir_str: str,
            name: str,
            language: str,
            ref_audio_path: str,
            ref_text: str,
            xvec_only: bool,
        ):
            if not name.strip():
                raise gr.Error("Profile name is required.")
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise gr.Error("Reference audio is required.")
            if (not xvec_only) and (not ref_text.strip()):
                raise gr.Error("Reference transcript is required unless x_vector_only_mode is enabled.")
            pid = safe_profile_id()
            dest_dir = profile_dir(Path(library_dir_str), pid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            ref_ext = Path(ref_audio_path).suffix or ".wav"
            ref_filename = f"ref_audio{ref_ext}"
            shutil.copy2(ref_audio_path, dest_dir / ref_filename)
            vp = VoiceProfile(
                profile_id=pid,
                name=name.strip(),
                task_type="Base",
                origin="Clone(Base)",
                created_at=now_iso(),
                language=language,
                voice="Vivian",
                instructions="",
                ref_text=ref_text.strip(),
                x_vector_only_mode=bool(xvec_only),
                ref_audio_filename=ref_filename,
            )
            save_profile(Path(library_dir_str), vp)
            return f"✅ Saved clone profile: {pid}"

        def on_library_refresh(library_dir_str: str):
            profiles = list_profiles(Path(library_dir_str))
            table = profiles_table_rows(profiles)
            choices = [p.profile_id for p in profiles]
            return table, gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        def on_load_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            vp = load_profile(Path(library_dir_str), pid.strip())
            ref_path = ""
            if vp.ref_audio_filename:
                candidate = profile_dir(Path(library_dir_str), vp.profile_id) / vp.ref_audio_filename
                if candidate.exists():
                    ref_path = str(candidate)
            return vp.__dict__, ref_path

        def on_delete_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            delete_profile(Path(library_dir_str), pid.strip())
            return "✅ Deleted profile."

        def on_export_selected(library_dir_str: str, pid: str):
            if not pid.strip():
                raise gr.Error("Provide a profile id.")
            zip_path = export_profiles_zip(Path(library_dir_str), profile_ids=[pid.strip()])
            return zip_path, f"✅ Exported: {Path(zip_path).name}"

        def on_export_all(library_dir_str: str):
            zip_path = export_profiles_zip(Path(library_dir_str), profile_ids=None)
            return zip_path, f"✅ Exported: {Path(zip_path).name}"

        def on_import_zip(library_dir_str: str, zip_file_path: Optional[str]):
            if not zip_file_path:
                raise gr.Error("Choose a ZIP file first.")
            summary = import_profiles_zip(Path(library_dir_str), Path(zip_file_path))
            msg = f"✅ Imported {summary['imported']} profile(s); skipped {summary['skipped']}."
            if summary["errors"]:
                msg += f" Issues: {'; '.join(summary['errors'][:3])}"
                if len(summary["errors"]) > 3:
                    msg += f" (+{len(summary['errors']) - 3} more)"
            return msg

        def on_play_generate(
            base_url: str,
            timeout_s: float,
            selected_model: Optional[str],
            library_dir_str: str,
            pid: str,
            text: str,
            fmt: str,
            speed: float,
            seed: float,
        ):
            active_model = ensure_backend_model(base_url, selected_model)
            if not pid:
                raise gr.Error("Pick a saved profile.")
            vp = load_profile(Path(library_dir_str), pid)
            payload: Dict[str, Any] = {
                "input": text,
                "response_format": fmt,
                "speed": float(speed),
                "language": vp.language,
            }
            if vp.task_type == "CustomVoice":
                payload.update({
                    "task_type": "CustomVoice",
                    "voice": vp.voice,
                    "instructions": vp.instructions or "",
                })
                # Use non-streaming for CustomVoice
                audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                return out_path, out_path, ""
            elif vp.task_type == "Base":
                payload.update({
                    "task_type": "Base",
                    "voice": vp.voice or "Vivian",
                    "x_vector_only_mode": bool(vp.x_vector_only_mode),
                    "seed": int(seed) if seed is not None else -1,
                    "cache_key": vp.profile_id,
                })
                if vp.ref_audio_filename:
                    ref_file = profile_dir(Path(library_dir_str), vp.profile_id) / vp.ref_audio_filename
                    if not ref_file.exists():
                        raise gr.Error("This profile is missing its reference audio file.")
                    payload["ref_audio"] = base64_from_file(ref_file)
                else:
                    raise gr.Error("This Base profile has no stored ref_audio.")
                if not vp.x_vector_only_mode:
                    if not vp.ref_text.strip():
                        raise gr.Error("This profile needs ref_text unless x_vector_only_mode is enabled.")
                    payload["ref_text"] = vp.ref_text.strip()
                
                # Use streaming for Base model (voice cloning)
                try:
                    audio_bytes, ext, timing_info = request_tts_streaming(base_url, payload, float(timeout_s))
                    out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                    
                    # Format timing info as readable markdown (include seed used)
                    first_chunk = timing_info.get('first_chunk_time') or 0.0
                    total_time = timing_info.get('total_time') or 0.0
                    audio_duration = timing_info.get('audio_duration') or 0.0
                    rtf = timing_info.get('rtf') or 0.0
                    chunk_count = timing_info.get('chunk_count', 0)
                    seed_used = timing_info.get('seed_used')
                    seed_line = f"| **Seed used** | {seed_used} |\n" if seed_used is not None else ""
                    
                    timing_md = f"""### ⏱️ Generation Timing
| Metric | Value |
|--------|-------|
| **First chunk** | {first_chunk:.2f}s |
| **Total time** | {total_time:.2f}s |
| **Audio duration** | {audio_duration:.2f}s |
| **RTF** | {rtf:.2f}x |
| **Chunks** | {chunk_count} |
| **Model** | {active_model or "current"} |
{seed_line}"""
                    return out_path, out_path, timing_md
                except Exception as e:
                    # Fallback to non-streaming voice clone endpoint
                    audio_bytes, ext, headers = request_tts_voice_clone(base_url, payload, float(timeout_s))
                    out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                    seed_used = headers.get("x-tts-seed") or headers.get("X-TTS-Seed")
                    extra = f" Seed used: {seed_used}." if seed_used else ""
                    return out_path, out_path, f"⚠️ Used non-streaming fallback: {e}{extra}"
            else:
                payload.update({
                    "task_type": "VoiceDesign",
                    "voice": vp.voice or "Vivian",
                    "instructions": vp.instructions or "",
                })
                audio_bytes, ext = request_tts(base_url, payload, float(timeout_s))
                out_path = write_bytes_to_temp_audio(audio_bytes, ext)
                return out_path, out_path, ""

        # ------------------------------------------------------------------
        # Wire up UI interactions to callbacks
        # ------------------------------------------------------------------

        refresh_voices_btn.click(
            fn=on_refresh_voices,
            inputs=[base_url_in, timeout_in],
            outputs=[state_base_url, state_voices, preset_voice, voices_status],
        )
        refresh_voices_btn.click(
            fn=fetch_backend_status,
            inputs=[base_url_in],
            outputs=[backend_status_md],
        )

        preset_generate_btn.click(
            fn=on_generate_preset,
            inputs=[base_url_in, timeout_in, preset_voice, preset_language, preset_instructions, preset_test_text],
            outputs=[preset_audio, preset_download, global_log],
        )
        preset_save_btn.click(
            fn=on_save_preset,
            inputs=[library_dir_in, preset_name, preset_voice, preset_language, preset_instructions],
            outputs=[global_log],
        )

        design_generate_btn.click(
            fn=on_generate_design_ref,
            inputs=[base_url_in, timeout_in, design_language, design_instructions, design_ref_line],
            outputs=[design_audio, design_download, global_log],
        )
        design_save_as_clone_btn.click(
            fn=on_save_design_as_clone,
            inputs=[library_dir_in, design_name, design_language, design_instructions, design_ref_line, design_audio],
            outputs=[global_log],
        )

        clone_generate_btn.click(
            fn=on_generate_clone,
            inputs=[base_url_in, timeout_in, backend_model_dropdown, clone_language, clone_ref_audio, clone_ref_text, clone_xvec_only, clone_test_text],
            outputs=[clone_audio, clone_download, global_log],
        )
        clone_save_btn.click(
            fn=on_save_clone_profile,
            inputs=[library_dir_in, clone_name, clone_language, clone_ref_audio, clone_ref_text, clone_xvec_only],
            outputs=[global_log],
        )

        library_refresh_btn.click(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id],
        )
        library_table.select(
            fn=on_library_table_select,
            inputs=[library_table],
            outputs=[selected_id],
        )
        load_selected_btn.click(
            fn=on_load_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[profile_details, ref_preview],
        )
        delete_selected_btn.click(
            fn=on_delete_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id],
        )

        export_selected_btn.click(
            fn=on_export_selected,
            inputs=[library_dir_in, selected_id],
            outputs=[export_file, global_log],
        )
        export_all_btn.click(
            fn=on_export_all,
            inputs=[library_dir_in],
            outputs=[export_file, global_log],
        )

        play_generate_btn.click(
            fn=on_play_generate,
            inputs=[base_url_in, timeout_in, backend_model_dropdown, library_dir_in, play_profile_id, play_text, play_response_format, play_speed, play_seed],
            outputs=[play_audio, play_download, play_timing],
        )

        demo.load(fn=on_library_refresh, inputs=[library_dir_in], outputs=[library_table, play_profile_id])
        demo.load(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        switch_model_btn.click(
            fn=do_switch_backend_model,
            inputs=[base_url_in, backend_model_dropdown],
            outputs=[backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        unload_model_btn.click(
            fn=do_unload_backend_model,
            inputs=[base_url_in],
            outputs=[backend_models_status_md, play_generate_btn, clone_generate_btn],
        ).then(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        refresh_model_status_btn.click(
            fn=load_backend_models,
            inputs=[base_url_in],
            outputs=[backend_model_column, backend_model_dropdown, backend_models_status_md, play_generate_btn, clone_generate_btn],
        )
        import_zip_btn.click(
            fn=on_import_zip,
            inputs=[library_dir_in, import_zip_file],
            outputs=[global_log],
        ).then(
            fn=on_library_refresh,
            inputs=[library_dir_in],
            outputs=[library_table, play_profile_id],
        )

    return demo


def main():
    """Launch the Voice Studio as a standalone web app for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_TTS_BASE_URL)
    parser.add_argument("--library-dir", default=str(DEFAULT_LIBRARY_DIR))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app(args.base_url, Path(args.library_dir))
    app.queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
