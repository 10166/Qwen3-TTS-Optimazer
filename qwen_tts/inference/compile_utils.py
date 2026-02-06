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

import logging

import torch

logger = logging.getLogger(__name__)


def apply_torch_compile(model, backend="inductor", mode="reduce-overhead"):
    """Apply torch.compile to compatible sub-components of a Qwen3TTSForConditionalGeneration model.

    Compiled components:
      - talker.code_predictor.model (code predictor backbone)
      - speech_tokenizer decoder.forward (Conv + Transformer + Upsample)

    Not compiled (would cause graph breaks or negligible speedup):
      - talker.model — main backbone, compiled via other optimization paths
      - speaker_encoder — small relative to total inference time
      - talker.text_projection / talker.codec_head — too small to benefit

    Args:
        model: Qwen3TTSForConditionalGeneration instance.
        backend: torch.compile backend (default: "inductor").
        mode: torch.compile mode (default: "reduce-overhead").
    """
    # Code predictor backbone — small transformer, called every generation step.
    # Use mode="default" (no CUDA Graphs) because prefill (seq_len=2) and decode
    # (seq_len=1) have different shapes, causing CUDA graph tensor overwrite errors
    # with reduce-overhead/max-autotune.
    model.talker.code_predictor.model = torch.compile(
        model.talker.code_predictor.model, backend=backend, mode="default"
    )
    logger.info("Compiled talker.code_predictor.model with backend=%s, mode=default", backend)

    # Tokenizer decoder forward — Conv + Transformer + Upsample
    # Only compile forward(), not chunked_decode() which has a while loop
    if hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None:
        decoder = model.speech_tokenizer.model.decoder
        decoder.forward = torch.compile(decoder.forward, backend=backend, mode=mode)
        logger.info("Compiled speech_tokenizer decoder.forward")
