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

"""Custom generation loop for continuous batching TTS inference.

Bypasses HuggingFace GenerationMixin.generate() by directly calling the backbone
(talker.model) and managing the generation loop manually. This enables:
  - Per-request KV cache management
  - Dynamic request insertion/removal during generation
  - Batched decode with different-length sequences
"""

import logging
from typing import List, Optional

import torch
from transformers.cache_utils import DynamicCache

from .kv_cache_manager import PerRequestKVCacheManager
from .request import RequestState, RequestStatus

logger = logging.getLogger(__name__)


class TTSGenerationLoop:
    """Custom generation loop that calls talker.model() directly."""

    def __init__(self, model):
        """
        Args:
            model: Qwen3TTSForConditionalGeneration instance.
        """
        self.model = model
        self.talker = model.talker
        self.config = model.config
        self.talker_config = model.config.talker_config
        self.kv_cache_manager = PerRequestKVCacheManager()

    @torch.inference_mode()
    def prefill_single(self, state: RequestState) -> None:
        """Run prefill for a single request.

        Steps:
          1. Build attention mask (all 1s, no padding for single request)
          2. Compute 3D position_ids and rope_deltas via get_rope_index
          3. Forward through talker.model (backbone)
          4. Apply codec_head to get logits → sample codebook-0
          5. Run code_predictor to generate codebooks 1..Q-1
          6. Store KV cache, past_hidden, rope_deltas, first codec_ids

        Args:
            state: Request state with prefill_embeds populated.
        """
        state.status = RequestStatus.PREFILLING
        device = state.prefill_embeds.device

        inputs_embeds = state.prefill_embeds  # (1, prefill_len, D)
        seq_len = inputs_embeds.shape[1]

        # Attention mask: all 1s (no padding for single request)
        attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

        # Compute 3D position_ids and rope_deltas
        position_ids, rope_deltas = self.talker.get_rope_index(attention_mask)
        # Adjust rope_deltas for no left-padding: delta0 = 0
        # In the original code: delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        # With no padding, delta0 = 0, so rope_deltas remain as-is but we subtract delta0 anyway
        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)  # = 0
        rope_deltas = rope_deltas - delta0

        # Create fresh KV cache
        kv_cache = DynamicCache()

        # Forward through backbone
        outputs = self.talker.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )

        hidden_states = outputs.last_hidden_state  # (1, seq_len, D)
        logits = self.talker.codec_head(hidden_states)  # (1, seq_len, vocab)

        # Sample codebook-0 from the last position
        last_logits = logits[:, -1, :]  # (1, vocab)
        first_token = self._sample_token(last_logits, state.request, state)

        # Run code_predictor to get codebooks 1..Q-1
        past_hidden = hidden_states[:, -1:, :]  # (1, 1, D)
        codec_ids = self._run_code_predictor(first_token, past_hidden, state)

        # Store state
        state.talker_kv_cache = outputs.past_key_values
        state.past_hidden = past_hidden
        state.rope_deltas = rope_deltas
        state.cache_seq_length = seq_len
        state.generation_step = 0
        state.num_generated_tokens = 1
        state.generated_codec_ids.append(codec_ids)
        state.status = RequestStatus.GENERATING

        # Check if first token is EOS
        if first_token.item() == state.eos_token_id:
            state.status = RequestStatus.COMPLETED

    @torch.inference_mode()
    def decode_step_batched(self, active_states: List[RequestState]) -> None:
        """Run one batched decode step for all active (GENERATING) requests.

        Steps:
          1. Build input embedding (seq_len=1) for each request
          2. Assemble batched KV cache
          3. Build attention mask (B, max_cache_len + 1)
          4. Build 3D position_ids (3, B, 1)
          5. Forward through talker.model batched
          6. Sample codebook-0 per request
          7. Run code_predictor per request for codebooks 1..Q-1
          8. Check EOS / max_tokens
          9. Disassemble batched KV cache

        Args:
            active_states: List of RequestState with status == GENERATING.
        """
        if not active_states:
            return

        batch_size = len(active_states)
        device = active_states[0].past_hidden.device

        # --- Step 1: Build input embeddings ---
        input_embeds_list = []
        for state in active_states:
            # Get last codec_ids and build combined embedding
            last_codec_ids = state.generated_codec_ids[-1]  # (num_code_groups,)
            embed = self._build_decode_input_embed(state, last_codec_ids)
            input_embeds_list.append(embed)

        inputs_embeds = torch.cat(input_embeds_list, dim=0)  # (B, 1, D)

        # --- Step 2: Assemble batched KV cache ---
        max_cache_len = max(s.cache_seq_length for s in active_states)
        batched_cache = self.kv_cache_manager.assemble_batch_cache(active_states, max_cache_len)

        # --- Step 3: Build attention mask ---
        # Shape: (B, max_cache_len + 1) — cache positions + current token
        attention_mask = torch.zeros(batch_size, max_cache_len + 1, device=device, dtype=torch.long)
        for i, state in enumerate(active_states):
            # Valid cache positions
            attention_mask[i, :state.cache_seq_length] = 1
            # Current token position
            attention_mask[i, max_cache_len] = 1

        # --- Step 4: Build 3D position_ids ---
        # Each request: position = cache_seq_length + rope_deltas
        position_ids = torch.zeros(3, batch_size, 1, device=device, dtype=torch.long)
        for i, state in enumerate(active_states):
            pos = state.cache_seq_length + state.rope_deltas  # rope_deltas: (1, 1)
            position_ids[:, i, :] = pos.squeeze()

        # --- Step 5: Forward through backbone ---
        # cache_position is computed automatically from batched_cache._seen_tokens (= max_cache_len)
        outputs = self.talker.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=batched_cache,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )

        hidden_states = outputs.last_hidden_state  # (B, 1, D)
        logits = self.talker.codec_head(hidden_states)  # (B, 1, vocab)

        # --- Step 6: Sample codebook-0 per request ---
        last_logits = logits[:, -1, :]  # (B, vocab)
        first_tokens = []
        for i, state in enumerate(active_states):
            token = self._sample_token(last_logits[i: i + 1], state.request, state)
            first_tokens.append(token)

        # --- Step 7: Run code_predictor per request ---
        for i, state in enumerate(active_states):
            past_hidden_i = hidden_states[i: i + 1, -1:, :]  # (1, 1, D)
            codec_ids = self._run_code_predictor(first_tokens[i], past_hidden_i, state)
            state.past_hidden = past_hidden_i
            state.generated_codec_ids.append(codec_ids)
            state.num_generated_tokens += 1
            state.generation_step += 1

        # --- Step 8: Check EOS / max_tokens ---
        for i, state in enumerate(active_states):
            first_token_val = first_tokens[i].item()
            if first_token_val == state.eos_token_id:
                state.status = RequestStatus.COMPLETED
            elif state.num_generated_tokens >= state.request.max_new_tokens:
                state.status = RequestStatus.COMPLETED

        # --- Step 9: Disassemble batched KV cache ---
        self.kv_cache_manager.disassemble_batch_cache(
            outputs.past_key_values, active_states, max_cache_len=max_cache_len, new_tokens=1
        )

    def _build_decode_input_embed(
        self, state: RequestState, last_codec_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build the input embedding for one decode step.

        Combines all codebook embeddings from last step + trailing text hidden.

        Args:
            state: Current request state.
            last_codec_ids: (num_code_groups,) tensor of codec IDs from last step.

        Returns:
            (1, 1, D) input embedding tensor.
        """
        num_groups = self.talker_config.num_code_groups

        # Embed codebook-0 using talker's main embedding
        embed_0 = self.talker.get_input_embeddings()(last_codec_ids[0:1].unsqueeze(0))  # (1, 1, D)

        # Embed codebooks 1..Q-1 using code_predictor's per-layer embeddings
        code_embeds = [embed_0]
        for g in range(1, num_groups):
            emb_layer = self.talker.code_predictor.get_input_embeddings()[g - 1]
            code_embeds.append(emb_layer(last_codec_ids[g: g + 1].unsqueeze(0)))  # (1, 1, D_cp)

        # Sum all codec embeddings
        # Note: talker embedding dim == code_predictor embedding dim in this architecture
        combined = sum(code_embeds)  # (1, 1, D)

        # Add trailing text hidden or tts_pad_embed
        gen_step = state.generation_step
        trailing = state.trailing_text_hidden  # (1, trailing_len, D)
        if gen_step < trailing.shape[1]:
            combined = combined + trailing[:, gen_step: gen_step + 1, :]
        else:
            combined = combined + state.tts_pad_embed

        return combined

    def _run_code_predictor(
        self, first_token: torch.Tensor, past_hidden: torch.Tensor, state: RequestState
    ) -> torch.Tensor:
        """Run code_predictor to generate codebooks 1..Q-1.

        Uses the HF generate() for the code_predictor since it's a short sequence
        (Q-1 steps) and doesn't need continuous batching.

        Args:
            first_token: (1, 1) codebook-0 token ID.
            past_hidden: (1, 1, D) hidden state from talker backbone.
            state: Request state for sampling parameters.

        Returns:
            (num_code_groups,) tensor of all codec IDs for this step.
        """
        req = state.request
        num_groups = self.talker_config.num_code_groups

        # Build input: [past_hidden, codebook-0_embed]
        first_embed = self.talker.get_input_embeddings()(first_token)  # (1, 1, D)
        cp_input = torch.cat([past_hidden, first_embed], dim=1)  # (1, 2, D)

        # Generate Q-1 tokens
        predictor_result = self.talker.code_predictor.generate(
            inputs_embeds=cp_input,
            max_new_tokens=num_groups - 1,
            do_sample=req.subtalker_dosample,
            top_p=req.subtalker_top_p,
            top_k=req.subtalker_top_k,
            temperature=req.subtalker_temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Combine: codebook-0 + codebooks 1..Q-1
        remaining_ids = predictor_result.sequences[:, -num_groups + 1:]  # (1, Q-1)
        all_ids = torch.cat([first_token, remaining_ids], dim=-1)  # (1, Q)

        return all_ids.squeeze(0)  # (Q,)

    def _sample_token(
        self,
        logits: torch.Tensor,
        request,
        state: RequestState,
    ) -> torch.Tensor:
        """Sample a single token from logits with per-request parameters.

        Applies: suppress_tokens → repetition_penalty → temperature → top_k → top_p → multinomial.

        Args:
            logits: (1, vocab_size) raw logits.
            request: TTSRequest with sampling parameters.
            state: RequestState for repetition penalty context.

        Returns:
            (1, 1) sampled token ID.
        """
        logits = logits.clone()

        # Suppress tokens
        if state.suppress_tokens:
            logits[:, state.suppress_tokens] = float("-inf")

        # Repetition penalty
        if request.repetition_penalty != 1.0 and state.generated_codec_ids:
            # Collect all previously generated codebook-0 tokens
            prev_tokens = torch.tensor(
                [ids[0].item() for ids in state.generated_codec_ids],
                device=logits.device,
                dtype=torch.long,
            )
            for token_id in prev_tokens.unique():
                score = logits[0, token_id]
                if score > 0:
                    logits[0, token_id] = score / request.repetition_penalty
                else:
                    logits[0, token_id] = score * request.repetition_penalty

        if not request.do_sample:
            return logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Temperature
        if request.temperature > 0 and request.temperature != 1.0:
            logits = logits / request.temperature

        # Top-k
        if request.top_k > 0:
            top_k = min(request.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p
        if request.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > request.top_p
            # Shift so that first token above threshold is also kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        return token
