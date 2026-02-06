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

"""Extract single-request embedding construction logic from Qwen3TTSForConditionalGeneration.generate().

This mirrors modeling_qwen3_tts.py lines 2068-2234, but operates on a single TTSRequest
instead of batched lists. The output (prefill_embeds, trailing_text_hidden, tts_pad_embed)
is exactly what the talker backbone expects as inputs_embeds for prefill.
"""

from typing import Tuple

import torch

from .request import GenerationMode, TTSRequest


class EmbeddingBuilder:
    """Builds prefill embeddings for a single TTSRequest."""

    def __init__(self, model, processor):
        """
        Args:
            model: Qwen3TTSForConditionalGeneration instance.
            processor: Qwen3TTSProcessor instance.
        """
        self.model = model
        self.talker = model.talker
        self.config = model.config
        self.processor = processor
        self.device = model.device
        self.dtype = model.dtype

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and return input_ids tensor of shape (1, seq_len)."""
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        return input_ids

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def build_request_embeddings(
        self,
        request: TTSRequest,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build prefill embeddings for a single request.

        Replicates the per-sample logic from Qwen3TTSForConditionalGeneration.generate()
        (modeling_qwen3_tts.py:2068-2234).

        Args:
            request: The TTS request to build embeddings for.

        Returns:
            prefill_embeds: (1, prefill_len, D) — concatenated input embeddings for prefill.
            trailing_text_hidden: (1, trailing_len, D) — text hidden states to add during decode.
            tts_pad_embed: (1, 1, D) — padding embedding for decode steps beyond trailing text.
        """
        # Tokenize the main text
        input_id = self._tokenize(self._build_assistant_text(request.text))

        # --- Resolve speaker embedding ---
        speaker_embed = self._resolve_speaker_embed(request)

        # --- Resolve language ID ---
        language_id = self._resolve_language_id(request)

        # --- Build tts_bos, tts_eos, tts_pad embeddings ---
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(
                torch.tensor(
                    [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                    device=self.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)  # 3 × (1, 1, D)

        # --- Build codec prefix embedding ---
        codec_input_embedding = self._build_codec_prefix(input_id, language_id, speaker_embed)

        # --- Build role prefix: <|im_start|>assistant\n ---
        role_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(input_id[:, :3])
        )

        # --- tts_pad * (prefix_len - 2) + tts_bos ---
        prefix_embed = torch.cat(
            (
                tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                tts_bos_embed,
            ),
            dim=1,
        ) + codec_input_embedding[:, :-1]

        talker_input_embed = torch.cat((role_embed, prefix_embed), dim=1)

        # --- Handle ICL (voice clone) vs standard text ---
        if (
            request.mode == GenerationMode.VOICE_CLONE
            and request.voice_clone_prompt is not None
            and request.voice_clone_prompt.get("ref_code") is not None
            and request.voice_clone_prompt.get("icl_mode", False)
        ):
            ref_id = self._tokenize(self._build_ref_text(request.ref_text))
            icl_input_embed, trailing_text_hidden = self.model.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=request.voice_clone_prompt["ref_code"].to(self.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=request.non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        else:
            # Standard text: first text token + codec_bos
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 3:4])
                    )
                    + codec_input_embedding[:, -1:],
                ],
                dim=1,
            )

            if request.non_streaming_mode:
                talker_input_embed = talker_input_embed[:, :-1]
                text_part = input_id[:, 3:-5]
                text_len = text_part.shape[1]
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        torch.cat(
                            (
                                self.talker.text_projection(
                                    self.talker.get_text_embeddings()(text_part)
                                ),
                                tts_eos_embed,
                            ),
                            dim=1,
                        )
                        + self.talker.get_input_embeddings()(
                            torch.tensor(
                                [[self.config.talker_config.codec_pad_id] * (text_len + 1)],
                                device=self.device,
                                dtype=input_id.dtype,
                            )
                        ),
                        tts_pad_embed
                        + self.talker.get_input_embeddings()(
                            torch.tensor(
                                [[self.config.talker_config.codec_bos_id]],
                                device=self.device,
                                dtype=input_id.dtype,
                            )
                        ),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                # Streaming mode: remaining text tokens + eos
                trailing_text_hidden = torch.cat(
                    (
                        self.talker.text_projection(
                            self.talker.get_text_embeddings()(input_id[:, 4:-5])
                        ),
                        tts_eos_embed,
                    ),
                    dim=1,
                )

        # --- Handle instruct prefix (prepend to talker_input_embed) ---
        instruct_embed = self._build_instruct_embed(request, input_id)
        if instruct_embed is not None:
            talker_input_embed = torch.cat([instruct_embed, talker_input_embed], dim=1)

        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    def _resolve_speaker_embed(self, request: TTSRequest):
        """Resolve speaker embedding based on request mode and parameters."""
        if request.mode == GenerationMode.VOICE_CLONE and request.voice_clone_prompt is not None:
            vcp = request.voice_clone_prompt
            if vcp.get("x_vector_only_mode", False) or vcp.get("icl_mode", False):
                ref_spk_embedding = vcp["ref_spk_embedding"]
                if isinstance(ref_spk_embedding, torch.Tensor):
                    return ref_spk_embedding.to(self.device).to(self.dtype)
            return None

        speaker = request.speaker
        if speaker is None or speaker == "":
            return None

        if speaker.lower() not in self.config.talker_config.spk_id:
            raise ValueError(f"Speaker {speaker} not supported")

        spk_id = self.config.talker_config.spk_id[speaker.lower()]
        return self.talker.get_input_embeddings()(
            torch.tensor(spk_id, device=self.device, dtype=torch.long)
        )

    def _resolve_language_id(self, request: TTSRequest):
        """Resolve language ID from request."""
        language = request.language
        if language is None or language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in self.config.talker_config.codec_language_id:
                raise ValueError(f"Language {language} not supported")
            language_id = self.config.talker_config.codec_language_id[language.lower()]

        # Check for dialect override
        speaker = request.speaker
        if (
            language is not None
            and language.lower() in ["chinese", "auto"]
            and speaker is not None
            and speaker != ""
            and self.config.talker_config.spk_is_dialect is not None
            and speaker.lower() in self.config.talker_config.spk_is_dialect
            and self.config.talker_config.spk_is_dialect[speaker.lower()] is not False
        ):
            dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
            language_id = self.config.talker_config.codec_language_id[dialect]

        return language_id

    def _build_codec_prefix(self, input_id, language_id, speaker_embed):
        """Build the codec prefix embedding (think/nothink + language + speaker + pad + bos)."""
        if language_id is None:
            codec_prefill_list = [[
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                self.config.talker_config.codec_think_id,
                self.config.talker_config.codec_think_bos_id,
                language_id,
                self.config.talker_config.codec_think_eos_id,
            ]]

        codec_input_embedding_0 = self.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=self.device, dtype=input_id.dtype)
        )
        codec_input_embedding_1 = self.talker.get_input_embeddings()(
            torch.tensor(
                [[self.config.talker_config.codec_pad_id, self.config.talker_config.codec_bos_id]],
                device=self.device,
                dtype=input_id.dtype,
            )
        )

        if speaker_embed is None:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, codec_input_embedding_1], dim=1
            )
        else:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1],
                dim=1,
            )

        return codec_input_embedding

    def _build_instruct_embed(self, request: TTSRequest, input_id: torch.Tensor):
        """Build instruct embedding if applicable."""
        instruct = request.instruct
        if instruct is None or instruct == "":
            return None

        instruct_id = self._tokenize(self._build_instruct_text(instruct))
        return self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))

