"""Qwen3-TTS FastAPI server with continuous batching support."""

import argparse
import asyncio
import hashlib
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from qwen_tts.inference.continuous_batching import (
    ContinuousBatchingEngine,
    GenerationMode,
    TTSRequest,
)
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem

logger = logging.getLogger(__name__)

# Speaker descriptions for CustomVoice model (9 speakers)
SPEAKER_DESCRIPTIONS: Dict[str, str] = {
    "vivian": "A warm and gentle female voice, suitable for storytelling and narration.",
    "aurora": "A bright and energetic female voice, suitable for cheerful content.",
    "serena": "A calm and soothing female voice, suitable for meditation and relaxation.",
    "maple": "A natural and friendly female voice, suitable for daily conversation.",
    "eden": "A deep and steady male voice, suitable for news broadcasting.",
    "felix": "A warm and magnetic male voice, suitable for audiobooks.",
    "river": "A clear and bright male voice, suitable for educational content.",
    "breeze": "A soft and gentle voice, suitable for whispering content.",
    "celeste": "A confident and professional female voice, suitable for business scenarios.",
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def numpy_to_wav_bytes(waveform: np.ndarray, sr: int) -> bytes:
    """Convert a numpy waveform to WAV bytes (PCM_16)."""
    buf = BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# VoiceCloneCache — LRU cache keyed by content hash
# ---------------------------------------------------------------------------


class VoiceCloneCache:
    """Caches VoiceClonePromptItems keyed by hash(audio + ref_text + x_vector_only_mode)."""

    def __init__(self, max_size: int = 100) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[VoiceClonePromptItem, float]] = {}
        self._max_size = max_size

    @staticmethod
    def compute_key(
        audio_identity: bytes,
        ref_text: Optional[str],
        x_vector_only_mode: bool,
    ) -> str:
        h = hashlib.sha256()
        h.update(audio_identity)
        h.update((ref_text or "").encode("utf-8"))
        h.update(b"\x01" if x_vector_only_mode else b"\x00")
        return h.hexdigest()[:16]

    def get(self, key: str) -> Optional[VoiceClonePromptItem]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            item, _ = entry
            self._cache[key] = (item, time.time())
            return item

    def set(self, key: str, item: VoiceClonePromptItem) -> None:
        with self._lock:
            if key not in self._cache and len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (item, time.time())

    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "keys": list(self._cache.keys()),
            }


# ---------------------------------------------------------------------------
# ModelRegistry — manages multiple model types
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Manages multiple Qwen3-TTS model instances and their engines."""

    def __init__(self) -> None:
        self.models: Dict[str, Qwen3TTSModel] = {}
        self.engines: Dict[str, ContinuousBatchingEngine] = {}
        self._preprocess_lock = threading.Lock()

    def load_model(
        self,
        model_type: str,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_batch_size: int = 16,
        enable_compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
    ) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        logger.info("Loading %s model from %s ...", model_type, model_path)
        wrapper = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        engine = ContinuousBatchingEngine(
            model=wrapper.model,
            processor=wrapper.processor,
            max_batch_size=max_batch_size,
            enable_compile=enable_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
        )

        if enable_compile:
            warmup_req = self._build_warmup_request(model_type)
            engine.warmup(warmup_req)

        engine.start_background()

        self.models[model_type] = wrapper
        self.engines[model_type] = engine
        logger.info("Loaded %s model successfully.", model_type)

    @staticmethod
    def _build_warmup_request(model_type: str) -> TTSRequest:
        """Build a minimal TTSRequest for warmup based on model type."""
        if model_type == "voice_design":
            return TTSRequest(
                request_id="__warmup__",
                mode=GenerationMode.VOICE_DESIGN,
                text="Warmup.",
                language="Auto",
                instruct="A warm voice.",
                max_new_tokens=10,
            )
        else:
            # custom_voice and base models share the same talker pipeline;
            # CUSTOM_VOICE mode without speaker works on all model types.
            return TTSRequest(
                request_id="__warmup__",
                mode=GenerationMode.CUSTOM_VOICE,
                text="Warmup.",
                language="Auto",
                max_new_tokens=10,
            )

    def get_engine(self, model_type: str) -> ContinuousBatchingEngine:
        engine = self.engines.get(model_type)
        if engine is None:
            raise HTTPException(status_code=503, detail=f"{model_type} model not loaded")
        return engine

    def get_model(self, model_type: str) -> Qwen3TTSModel:
        model = self.models.get(model_type)
        if model is None:
            raise HTTPException(status_code=503, detail=f"{model_type} model not loaded")
        return model

    def shutdown(self) -> None:
        for name, engine in self.engines.items():
            logger.info("Stopping engine for %s ...", name)
            engine.stop_background()
        logger.info("All engines stopped.")


# ---------------------------------------------------------------------------
# Application globals
# ---------------------------------------------------------------------------

registry = ModelRegistry()
clone_cache = VoiceCloneCache()
REQUEST_TIMEOUT: float = 300.0


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    registry.shutdown()


app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    models = {name: name for name in registry.engines}
    return {"ok": True, "models": models}


@app.get("/model_info")
def model_info():
    loaded_models: Dict[str, Any] = {}
    for name, wrapper in registry.models.items():
        speakers = wrapper.get_supported_speakers()
        languages = wrapper.get_supported_languages()
        loaded_models[name] = {
            "speakers": speakers,
            "languages": languages,
        }
    features = {
        "voice_design": "voice_design" in registry.engines,
        "custom_voice": "custom_voice" in registry.engines,
        "voice_clone": "base" in registry.engines,
    }
    return {"loaded_models": loaded_models, "features": features}


@app.get("/speakers")
def speakers():
    wrapper = registry.get_model("custom_voice")
    supported = wrapper.get_supported_speakers() or []
    result: List[Dict[str, str]] = []
    for spk in supported:
        desc = SPEAKER_DESCRIPTIONS.get(spk.lower(), "")
        result.append({"name": spk, "description": desc})
    return {"speakers": result}


# ---------------------------------------------------------------------------
# Clone cache endpoint
# ---------------------------------------------------------------------------


@app.get("/tts/clone_cache")
def get_clone_cache():
    return clone_cache.info()


# ---------------------------------------------------------------------------
# Core TTS helpers
# ---------------------------------------------------------------------------


async def _submit_and_wait(
    engine: ContinuousBatchingEngine,
    request: TTSRequest,
    cache_key: Optional[str] = None,
) -> Response:
    """Submit a TTSRequest to the engine and return a WAV response."""
    logger.info("Request %s [%s] text=%r", request.request_id, request.mode.value, request.text[:80])
    engine.add_request(request)
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    try:
        waveform, sr = await loop.run_in_executor(
            None, engine.wait_for_result, request.request_id, REQUEST_TIMEOUT
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Generation timed out")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    elapsed = time.monotonic() - t0
    logger.info("Request %s completed in %.2fs, audio=%.2fs", request.request_id, elapsed, len(waveform) / sr)

    headers: Dict[str, str] = {"Content-Disposition": "attachment; filename=output.wav"}
    if cache_key:
        headers["X-Cache-Key"] = cache_key

    wav_bytes = numpy_to_wav_bytes(waveform, sr)
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


def _voice_clone_prompt_item_to_dict(item: VoiceClonePromptItem) -> Dict[str, Any]:
    """Convert a single VoiceClonePromptItem to the dict format expected by TTSRequest.

    Note: ContinuousBatchingEngine / EmbeddingBuilder expects bare values (not lists),
    unlike the batched API in Qwen3TTSModel._prompt_items_to_voice_clone_prompt.
    """
    return {
        "ref_code": item.ref_code,
        "ref_spk_embedding": item.ref_spk_embedding,
        "x_vector_only_mode": item.x_vector_only_mode,
        "icl_mode": item.icl_mode,
    }


# ---------------------------------------------------------------------------
# Pydantic request models for JSON endpoints
# ---------------------------------------------------------------------------


class VoiceDesignBody(BaseModel):
    text: str
    language: str = "Auto"
    instruct: str


class CustomVoiceBody(BaseModel):
    text: str
    language: str = "Auto"
    speaker: Optional[str] = None


# ---------------------------------------------------------------------------
# Voice Design endpoint
# ---------------------------------------------------------------------------


@app.post("/tts/voice_design")
async def tts_voice_design(body: VoiceDesignBody):
    engine = registry.get_engine("voice_design")
    req = TTSRequest(
        request_id=str(uuid.uuid4()),
        mode=GenerationMode.VOICE_DESIGN,
        text=body.text,
        language=body.language,
        instruct=body.instruct,
    )
    return await _submit_and_wait(engine, req)


# ---------------------------------------------------------------------------
# Custom Voice endpoint
# ---------------------------------------------------------------------------


@app.post("/tts/custom_voice")
async def tts_custom_voice(body: CustomVoiceBody):
    engine = registry.get_engine("custom_voice")
    wrapper = registry.get_model("custom_voice")

    if body.speaker:
        supported = wrapper.get_supported_speakers()
        if supported and body.speaker.lower() not in [s.lower() for s in supported]:
            raise HTTPException(
                status_code=400,
                detail=f"Speaker '{body.speaker}' not supported. Available: {supported}",
            )

    req = TTSRequest(
        request_id=str(uuid.uuid4()),
        mode=GenerationMode.CUSTOM_VOICE,
        text=body.text,
        language=body.language,
        speaker=body.speaker,
    )
    return await _submit_and_wait(engine, req)


# ---------------------------------------------------------------------------
# Voice Clone endpoint
# ---------------------------------------------------------------------------


async def _build_clone_prompt(
    wrapper: Qwen3TTSModel,
    ref_audio: Any = None,
    ref_audio_file: Optional[UploadFile] = None,
    ref_text: Optional[str] = None,
    cache_key: Optional[str] = None,
    x_vector_only_mode: bool = False,
) -> Tuple[VoiceClonePromptItem, str]:
    """Build or retrieve a VoiceClonePromptItem. Returns (item, cache_key)."""

    # Direct cache_key lookup (e.g. reuse a previously returned X-Cache-Key)
    if cache_key:
        item = clone_cache.get(cache_key)
        if item is None:
            raise HTTPException(
                status_code=400,
                detail=f"Cache key '{cache_key}' not found or expired",
            )
        logger.info("Clone prompt cache hit (key=%s)", cache_key)
        return item, cache_key

    # ICL mode (x_vector_only_mode=False) requires ref_text
    if not x_vector_only_mode and not ref_text:
        raise HTTPException(
            status_code=400,
            detail="ref_text is required when x_vector_only_mode is false (ICL mode). "
            "Provide ref_text or set x_vector_only_mode=true.",
        )

    # Determine audio source & compute cache key from content
    audio_input: Any = None
    audio_identity: bytes
    if ref_audio_file is not None:
        raw = await ref_audio_file.read()
        audio_data, sr = sf.read(BytesIO(raw))
        audio_input = (audio_data, sr)
        audio_identity = raw
    elif ref_audio is not None:
        audio_input = ref_audio  # URL or base64 string
        audio_identity = ref_audio.encode("utf-8")
    else:
        raise HTTPException(
            status_code=400,
            detail="ref_audio, ref_audio_file, or cache_key is required",
        )

    key = clone_cache.compute_key(audio_identity, ref_text, x_vector_only_mode)

    # Check cache
    cached = clone_cache.get(key)
    if cached is not None:
        logger.info("Clone prompt cache hit (key=%s)", key)
        return cached, key

    # Cache miss — run expensive preprocessing
    loop = asyncio.get_running_loop()

    def _create_prompt():
        with registry._preprocess_lock:
            return wrapper.create_voice_clone_prompt(
                ref_audio=audio_input,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )

    try:
        items = await loop.run_in_executor(None, _create_prompt)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process reference audio: {exc}")

    item = items[0]
    clone_cache.set(key, item)
    logger.info("Clone prompt cached (key=%s)", key)
    return item, key


@app.post("/tts/clone")
async def tts_clone(
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_audio: Optional[str] = Form(None),
    ref_audio_file: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    cache_key: Optional[str] = Form(None),
    x_vector_only_mode: bool = Form(False),
):
    engine = registry.get_engine("base")
    wrapper = registry.get_model("base")

    item, key = await _build_clone_prompt(
        wrapper,
        ref_audio=ref_audio,
        ref_audio_file=ref_audio_file,
        ref_text=ref_text,
        cache_key=cache_key,
        x_vector_only_mode=x_vector_only_mode,
    )

    prompt_dict = _voice_clone_prompt_item_to_dict(item)
    req = TTSRequest(
        request_id=str(uuid.uuid4()),
        mode=GenerationMode.VOICE_CLONE,
        text=text,
        language=language,
        voice_clone_prompt=prompt_dict,
        ref_text=item.ref_text,
    )
    return await _submit_and_wait(engine, req, cache_key=key)


# ---------------------------------------------------------------------------
# Unified TTS endpoint
# ---------------------------------------------------------------------------


@app.post("/tts")
async def tts_unified(
    text: str = Form(...),
    language: str = Form("Auto"),
    mode: str = Form("custom_voice"),
    instruct: Optional[str] = Form(None),
    speaker: Optional[str] = Form(None),
    ref_audio: Optional[str] = Form(None),
    ref_audio_file: Optional[UploadFile] = File(None),
    ref_text: Optional[str] = Form(None),
    cache_key: Optional[str] = Form(None),
    x_vector_only_mode: bool = Form(False),
):
    if mode == "voice_design":
        if not instruct:
            raise HTTPException(status_code=400, detail="'instruct' is required for voice_design mode")

        engine = registry.get_engine("voice_design")
        req = TTSRequest(
            request_id=str(uuid.uuid4()),
            mode=GenerationMode.VOICE_DESIGN,
            text=text,
            language=language,
            instruct=instruct,
        )
        return await _submit_and_wait(engine, req)

    elif mode == "custom_voice":
        engine = registry.get_engine("custom_voice")
        wrapper = registry.get_model("custom_voice")

        if speaker:
            supported = wrapper.get_supported_speakers()
            if supported and speaker.lower() not in [s.lower() for s in supported]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Speaker '{speaker}' not supported. Available: {supported}",
                )

        req = TTSRequest(
            request_id=str(uuid.uuid4()),
            mode=GenerationMode.CUSTOM_VOICE,
            text=text,
            language=language,
            speaker=speaker,
        )
        return await _submit_and_wait(engine, req)

    elif mode == "voice_clone":
        engine = registry.get_engine("base")
        wrapper = registry.get_model("base")

        item, key = await _build_clone_prompt(
            wrapper,
            ref_audio=ref_audio,
            ref_audio_file=ref_audio_file,
            ref_text=ref_text,
            cache_key=cache_key,
            x_vector_only_mode=x_vector_only_mode,
        )

        prompt_dict = _voice_clone_prompt_item_to_dict(item)
        req = TTSRequest(
            request_id=str(uuid.uuid4()),
            mode=GenerationMode.VOICE_CLONE,
            text=text,
            language=language,
            voice_clone_prompt=prompt_dict,
            ref_text=item.ref_text,
        )
        return await _submit_and_wait(engine, req, cache_key=key)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: voice_design, custom_voice, voice_clone",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS FastAPI Server")
    parser.add_argument("--base-model", type=str, default=None, help="Path to Base model (voice clone)")
    parser.add_argument("--custom-voice-model", type=str, default=None, help="Path to CustomVoice model")
    parser.add_argument("--voice-design-model", type=str, default=None, help="Path to VoiceDesign model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", type=str, default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--compile-backend", type=str, default="inductor")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--request-timeout", type=float, default=300.0)

    args = parser.parse_args()

    if not any([args.base_model, args.custom_voice_model, args.voice_design_model]):
        parser.error("At least one model must be specified (--base-model, --custom-voice-model, or --voice-design-model)")

    global REQUEST_TIMEOUT
    REQUEST_TIMEOUT = args.request_timeout

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    common_kwargs = dict(
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_batch_size=args.max_batch_size,
        enable_compile=args.enable_compile,
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
    )

    if args.base_model:
        registry.load_model("base", args.base_model, **common_kwargs)
    if args.custom_voice_model:
        registry.load_model("custom_voice", args.custom_voice_model, **common_kwargs)
    if args.voice_design_model:
        registry.load_model("voice_design", args.voice_design_model, **common_kwargs)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
