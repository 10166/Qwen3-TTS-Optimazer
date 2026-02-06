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
      - talker.model (main transformer backbone)
      - talker.code_predictor.model (code predictor backbone)
      - speaker_encoder (ECAPA-TDNN, fullgraph=True, only for base model)
      - speech_tokenizer decoder.forward (Conv + Transformer + Upsample)
      - talker.text_projection (MLP)
      - talker.codec_head (Linear)

    Not compiled (would cause graph breaks):
      - Qwen3TTSTalkerForConditionalGeneration.forward() — calls code_predictor.generate()
      - Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward() — dynamic branches
      - chunked_decode() — while loop (but its inner self() call benefits from compiled decoder.forward)

    Args:
        model: Qwen3TTSForConditionalGeneration instance.
        backend: torch.compile backend (default: "inductor").
        mode: torch.compile mode (default: "reduce-overhead").
    """
    # Talker backbone — main transformer decoder, largest compute
    model.talker.model = torch.compile(model.talker.model, backend=backend, mode=mode)
    logger.info("Compiled talker.model (backbone) with backend=%s, mode=%s", backend, mode)

    # Code predictor backbone — small transformer, called every generation step
    model.talker.code_predictor.model = torch.compile(
        model.talker.code_predictor.model, backend=backend, mode=mode
    )
    logger.info("Compiled talker.code_predictor.model with backend=%s, mode=%s", backend, mode)

    # Speaker encoder (ECAPA-TDNN) — only exists for base model, no dynamic branches
    if model.speaker_encoder is not None:
        model.speaker_encoder = torch.compile(
            model.speaker_encoder, backend=backend, mode=mode, fullgraph=True
        )
        logger.info("Compiled speaker_encoder with fullgraph=True")

    # Tokenizer decoder forward — Conv + Transformer + Upsample
    # Only compile forward(), not chunked_decode() which has a while loop
    if hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None:
        decoder = model.speech_tokenizer.model.decoder
        decoder.forward = torch.compile(decoder.forward, backend=backend, mode=mode)
        logger.info("Compiled speech_tokenizer decoder.forward")

    # Small components — frequently called per step
    model.talker.text_projection = torch.compile(
        model.talker.text_projection, backend=backend, mode=mode
    )
    logger.info("Compiled talker.text_projection")

    model.talker.codec_head = torch.compile(
        model.talker.codec_head, backend=backend, mode=mode
    )
    logger.info("Compiled talker.codec_head")
