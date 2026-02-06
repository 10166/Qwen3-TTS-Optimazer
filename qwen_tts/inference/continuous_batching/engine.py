# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Continuous Batching Engine for Qwen3-TTS.

Provides dynamic request insertion/removal during generation, per-request KV cache
management, and early completion for short requests.
"""

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ...core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from ..compile_utils import apply_torch_compile
from .embedding_builder import EmbeddingBuilder
from .generation_loop import TTSGenerationLoop
from .request import GenerationMode, RequestState, RequestStatus, TTSRequest

logger = logging.getLogger(__name__)


class ContinuousBatchingEngine:
    """Continuous batching engine for Qwen3-TTS inference.

    Supports:
      - Dynamic request insertion (add_request) even while generation is running
      - Per-request KV cache management
      - Early completion: short requests finish and release resources immediately
      - Optional torch.compile for inference speedup
      - Background generation thread

    Usage:
        engine = ContinuousBatchingEngine.from_pretrained("Qwen/Qwen3-TTS-1.7B-CustomVoice")
        r1 = engine.add_request(TTSRequest(request_id="r1", mode=GenerationMode.CUSTOM_VOICE, ...))
        r2 = engine.add_request(TTSRequest(request_id="r2", mode=GenerationMode.CUSTOM_VOICE, ...))
        engine.run()
        wav1, sr = engine.get_result("r1")
        wav2, sr = engine.get_result("r2")
    """

    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        processor,
        max_batch_size: int = 16,
        enable_compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
    ):
        self.model = model
        self.processor = processor
        self.max_batch_size = max_batch_size

        if enable_compile:
            apply_torch_compile(model, backend=compile_backend, mode=compile_mode)

        self.embedding_builder = EmbeddingBuilder(model, processor)
        self.generation_loop = TTSGenerationLoop(model)

        # Request queues (thread-safe via lock)
        self._lock = threading.Lock()
        self._pending: OrderedDict[str, RequestState] = OrderedDict()
        self._active: OrderedDict[str, RequestState] = OrderedDict()
        self._completed: Dict[str, RequestState] = {}
        self._result_event: Dict[str, threading.Event] = {}

        # Background thread
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Build suppress_tokens list once
        talker_config = model.config.talker_config
        self._suppress_tokens = [
            i
            for i in range(talker_config.vocab_size - 1024, talker_config.vocab_size)
            if i != talker_config.codec_eos_token_id
        ]
        self._eos_token_id = talker_config.codec_eos_token_id

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        max_batch_size: int = 16,
        enable_compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "reduce-overhead",
        **model_kwargs,
    ) -> "ContinuousBatchingEngine":
        """Load model and create engine.

        Args:
            model_path: HuggingFace repo id or local directory.
            max_batch_size: Maximum concurrent requests in a batch.
            enable_compile: Whether to apply torch.compile.
            compile_backend: torch.compile backend.
            compile_mode: torch.compile mode.
            **model_kwargs: Forwarded to AutoModel.from_pretrained.

        Returns:
            ContinuousBatchingEngine instance.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(model_path, **model_kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration."
            )
        processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)

        return cls(
            model=model,
            processor=processor,
            max_batch_size=max_batch_size,
            enable_compile=enable_compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
        )

    def add_request(self, request: TTSRequest) -> str:
        """Submit a TTS request. Thread-safe, can be called during generation.

        Args:
            request: The TTS request to process.

        Returns:
            The request_id.
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        state = RequestState(
            request=request,
            status=RequestStatus.PENDING,
            suppress_tokens=self._suppress_tokens,
            eos_token_id=self._eos_token_id,
        )

        event = threading.Event()
        with self._lock:
            self._pending[request.request_id] = state
            self._result_event[request.request_id] = event

        return request.request_id

    def get_result(self, request_id: str) -> Optional[Tuple[np.ndarray, int]]:
        """Get and remove the result for a completed request.

        Args:
            request_id: The request ID.

        Returns:
            (waveform, sample_rate) if completed, None otherwise.
        """
        with self._lock:
            state = self._completed.pop(request_id, None)
            if state is not None:
                self._result_event.pop(request_id, None)
        if state is not None:
            return state.result_audio
        return None

    def is_complete(self, request_id: str) -> bool:
        """Check if a request has completed."""
        with self._lock:
            return request_id in self._completed

    def wait_for_result(self, request_id: str, timeout: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Block until the specified request completes.

        Args:
            request_id: The request to wait for.
            timeout: Maximum wait time in seconds.

        Returns:
            (waveform, sample_rate).

        Raises:
            TimeoutError: If timeout is exceeded.
            RuntimeError: If the request failed.
        """
        with self._lock:
            event = self._result_event.get(request_id)
        if event is None:
            raise ValueError(f"Unknown request: {request_id}")

        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Timed out waiting for request {request_id}")

        with self._lock:
            state = self._completed.pop(request_id, None)
            self._result_event.pop(request_id, None)
        if state is None:
            raise RuntimeError(f"Request {request_id} not found in completed")
        if state.status == RequestStatus.FAILED:
            raise RuntimeError(f"Request {request_id} failed")

        return state.result_audio

    def run(self) -> None:
        """Run the generation loop until all current requests complete.

        This is a blocking call. For async usage, use start_background() instead.
        """
        self._generation_loop()

    def start_background(self) -> None:
        """Start the generation loop in a background thread."""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_flag.clear()
        self._bg_thread = threading.Thread(target=self._background_loop, daemon=True)
        self._bg_thread.start()

    def stop_background(self) -> None:
        """Stop the background generation thread."""
        self._stop_flag.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=30)
            self._bg_thread = None

    def _background_loop(self) -> None:
        """Background thread target: continuously process requests."""
        while not self._stop_flag.is_set():
            with self._lock:
                has_work = bool(self._pending) or bool(self._active)
            if has_work:
                self._generation_loop()
            else:
                self._stop_flag.wait(timeout=0.01)

    def _generation_loop(self) -> None:
        """Main generation loop.

        while has_work:
          1. Admit pending requests (up to max_batch_size)
          2. Prefill new requests (one at a time)
          3. Batched decode step for all GENERATING requests
          4. Retire completed requests (decode codec → audio)
        """
        while True:
            # 1. Admit pending → active
            self._admit_pending_requests()

            # 2. Prefill new requests
            self._prefill_new_requests()

            # 3. Batched decode
            with self._lock:
                generating = [
                    s for s in self._active.values()
                    if s.status == RequestStatus.GENERATING
                ]

            if generating:
                self.generation_loop.decode_step_batched(generating)

            # 4. Retire completed
            self._retire_completed_requests()

            # Check if there's more work
            with self._lock:
                has_work = bool(self._pending) or bool(self._active)
            if not has_work:
                break

    def _admit_pending_requests(self) -> None:
        """Move pending requests to active, respecting max_batch_size."""
        with self._lock:
            slots = self.max_batch_size - len(self._active)
            if slots <= 0 or not self._pending:
                return

            to_admit = []
            for rid, state in list(self._pending.items()):
                if len(to_admit) >= slots:
                    break
                to_admit.append((rid, state))

            for rid, state in to_admit:
                del self._pending[rid]
                self._active[rid] = state

    def _prefill_new_requests(self) -> None:
        """Prefill requests that are in active but still PENDING status."""
        with self._lock:
            to_prefill = [
                s for s in self._active.values()
                if s.status == RequestStatus.PENDING
            ]

        for state in to_prefill:
            try:
                # Build embeddings
                prefill_embeds, trailing_text_hidden, tts_pad_embed = (
                    self.embedding_builder.build_request_embeddings(state.request)
                )
                state.prefill_embeds = prefill_embeds
                state.trailing_text_hidden = trailing_text_hidden
                state.tts_pad_embed = tts_pad_embed

                # Run prefill
                self.generation_loop.prefill_single(state)

                # Free prefill_embeds after prefill (no longer needed)
                state.prefill_embeds = None

            except Exception as e:
                logger.error("Prefill failed for %s: %s", state.request.request_id, e, exc_info=True)
                state.status = RequestStatus.FAILED
                self._mark_completed(state.request.request_id)

    def _retire_completed_requests(self) -> None:
        """Move completed/failed requests from active to completed, decode audio."""
        with self._lock:
            done_ids = [
                rid for rid, s in self._active.items()
                if s.status in (RequestStatus.COMPLETED, RequestStatus.FAILED)
            ]

        for rid in done_ids:
            with self._lock:
                state = self._active.pop(rid, None)
            if state is None:
                continue

            if state.status == RequestStatus.COMPLETED:
                try:
                    self._decode_audio(state)
                except Exception as e:
                    logger.error("Audio decode failed for %s: %s", rid, e, exc_info=True)
                    state.status = RequestStatus.FAILED

            with self._lock:
                self._completed[rid] = state
                event = self._result_event.get(rid)
            if event is not None:
                event.set()

    def _mark_completed(self, request_id: str) -> None:
        """Move request from active/pending to completed dict and signal event.

        Used for early failures (e.g. prefill error) where the request hasn't
        been popped from active yet.
        """
        with self._lock:
            state = self._active.pop(request_id, None)
            if state is None:
                state = self._pending.pop(request_id, None)
            if state is not None:
                self._completed[request_id] = state
            event = self._result_event.get(request_id)
        if event is not None:
            event.set()

    @torch.inference_mode()
    def _decode_audio(self, state: RequestState) -> None:
        """Decode generated codec IDs to audio waveform.

        Args:
            state: Completed request state with generated_codec_ids.
        """
        if not state.generated_codec_ids:
            state.result_audio = (np.array([], dtype=np.float32), 24000)
            return

        # Stack codec IDs: list of (Q,) → (T, Q)
        codec_ids = torch.stack(state.generated_codec_ids, dim=0)  # (T, Q)

        # Remove EOS tokens from the end
        first_codebook = codec_ids[:, 0]
        eos_mask = first_codebook == state.eos_token_id
        if eos_mask.any():
            eos_idx = torch.argmax(eos_mask.int())
            codec_ids = codec_ids[:eos_idx]

        if codec_ids.shape[0] == 0:
            state.result_audio = (np.array([], dtype=np.float32), 24000)
            return

        # Handle voice clone: prepend ref_code for decode, then trim
        ref_code = None
        if (
            state.request.mode == GenerationMode.VOICE_CLONE
            and state.request.voice_clone_prompt is not None
        ):
            ref_code = state.request.voice_clone_prompt.get("ref_code")
            if ref_code is not None:
                ref_code = ref_code.to(codec_ids.device)
                decode_codes = torch.cat([ref_code, codec_ids], dim=0)
            else:
                decode_codes = codec_ids
        else:
            decode_codes = codec_ids

        # Decode through speech tokenizer
        wavs, fs = self.model.speech_tokenizer.decode(
            [{"audio_codes": decode_codes}]
        )
        wav = wavs[0]

        # Trim reference audio portion for voice clone
        if ref_code is not None:
            ref_len = ref_code.shape[0]
            total_len = decode_codes.shape[0]
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wav = wav[cut:]

        state.result_audio = (wav, fs)

        # Free GPU memory
        state.talker_kv_cache = None
        state.generated_codec_ids = []
        state.past_hidden = None
        state.trailing_text_hidden = None
        state.tts_pad_embed = None
