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

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers.cache_utils import DynamicCache


class RequestStatus(Enum):
    PENDING = "pending"
    PREFILLING = "prefilling"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationMode(Enum):
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    VOICE_CLONE = "voice_clone"


@dataclass
class TTSRequest:
    """A single TTS generation request."""

    request_id: str
    mode: GenerationMode
    text: str
    language: str = "Auto"

    # CUSTOM_VOICE mode
    speaker: Optional[str] = None
    instruct: Optional[str] = None  # Also used for VOICE_DESIGN

    # VOICE_CLONE mode
    voice_clone_prompt: Optional[Dict[str, Any]] = None
    ref_text: Optional[str] = None

    non_streaming_mode: bool = False

    # Generation params
    max_new_tokens: int = 2048
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05

    # Code predictor (sub-talker) params
    subtalker_dosample: bool = True
    subtalker_top_k: int = 50
    subtalker_top_p: float = 1.0
    subtalker_temperature: float = 0.9


@dataclass
class RequestState:
    """Runtime state for a request during generation."""

    request: TTSRequest
    status: RequestStatus

    # Embeddings built during prefill setup
    prefill_embeds: Optional[torch.Tensor] = None      # (1, prefill_len, D)
    trailing_text_hidden: Optional[torch.Tensor] = None  # (1, trailing_len, D)
    tts_pad_embed: Optional[torch.Tensor] = None         # (1, 1, D)

    # Talker KV cache (per-request)
    talker_kv_cache: Optional[DynamicCache] = None

    # Generation tracking
    generation_step: int = 0
    num_generated_tokens: int = 0
    past_hidden: Optional[torch.Tensor] = None  # (1, 1, D)
    rope_deltas: Optional[torch.Tensor] = None
    cache_seq_length: int = 0

    # Accumulated codec IDs per step
    generated_codec_ids: List[torch.Tensor] = field(default_factory=list)

    # Final result
    result_audio: Optional[Tuple[np.ndarray, int]] = None

    # Suppress tokens list (computed once per request)
    suppress_tokens: Optional[List[int]] = None

    # EOS token ID for this request
    eos_token_id: Optional[int] = None
