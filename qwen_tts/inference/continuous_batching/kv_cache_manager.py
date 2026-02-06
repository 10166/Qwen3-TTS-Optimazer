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

from typing import List

import torch
from transformers.cache_utils import DynamicCache

from .request import RequestState


class PerRequestKVCacheManager:
    """Manages per-request KV caches and assembles/disassembles them for batched decode."""

    @staticmethod
    def create_cache() -> DynamicCache:
        return DynamicCache()

    @staticmethod
    def _set_seen_tokens(cache: DynamicCache, seq_length: int) -> None:
        """Set the internal _seen_tokens counter on a DynamicCache."""
        cache._seen_tokens = seq_length

    @staticmethod
    def assemble_batch_cache(
        states: List[RequestState],
        max_cache_len: int,
    ) -> DynamicCache:
        """Assemble per-request KV caches into a single batched DynamicCache.

        For each layer, concatenates K and V tensors along the batch dimension,
        right-padding shorter caches with zeros. The caller must construct an
        appropriate attention mask to ignore the padded positions.

        Args:
            states: Active request states with populated talker_kv_cache.
            max_cache_len: Maximum cache sequence length across all requests.

        Returns:
            A DynamicCache with batched K/V of shape (B, num_kv_heads, max_cache_len, head_dim).
        """
        if not states:
            return DynamicCache()

        num_layers = len(states[0].talker_kv_cache.key_cache)
        batched_cache = DynamicCache()

        for layer_idx in range(num_layers):
            keys = []
            values = []
            for state in states:
                k = state.talker_kv_cache.key_cache[layer_idx]    # (1, H, S_i, D)
                v = state.talker_kv_cache.value_cache[layer_idx]  # (1, H, S_i, D)
                seq_len = k.shape[2]
                if seq_len < max_cache_len:
                    pad_len = max_cache_len - seq_len
                    k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))  # pad seq dim
                    v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
                keys.append(k)
                values.append(v)

            batched_k = torch.cat(keys, dim=0)   # (B, H, max_cache_len, D)
            batched_v = torch.cat(values, dim=0)  # (B, H, max_cache_len, D)

            batched_cache.key_cache.append(batched_k)
            batched_cache.value_cache.append(batched_v)

        # Set _seen_tokens to max_cache_len so cache_position is computed correctly
        PerRequestKVCacheManager._set_seen_tokens(batched_cache, max_cache_len)

        return batched_cache

    @staticmethod
    def disassemble_batch_cache(
        batched_cache: DynamicCache,
        states: List[RequestState],
        max_cache_len: int,
        new_tokens: int = 1,
    ) -> None:
        """Split a batched DynamicCache back into per-request caches.

        After a batched forward pass, the cache has shape (B, H, max_cache_len + new_tokens, D).
        The new KV entries are appended at positions [max_cache_len, max_cache_len + new_tokens).
        For each request, we reconstruct a compact cache by taking:
          - Original valid positions: [0, cache_seq_length)
          - New positions: [max_cache_len, max_cache_len + new_tokens)

        Args:
            batched_cache: The batched cache after the forward pass.
            states: Active request states to update.
            max_cache_len: The max_cache_len used when assembling (before forward).
            new_tokens: Number of new tokens appended during the forward pass.
        """
        num_layers = len(batched_cache.key_cache)

        for i, state in enumerate(states):
            old_len = state.cache_seq_length
            new_len = old_len + new_tokens
            new_cache = DynamicCache()

            for layer_idx in range(num_layers):
                k_full = batched_cache.key_cache[layer_idx][i: i + 1]   # (1, H, max_cache_len + new_tokens, D)
                v_full = batched_cache.value_cache[layer_idx][i: i + 1]

                # Take valid old entries + new entries
                k_old = k_full[:, :, :old_len, :]
                v_old = v_full[:, :, :old_len, :]
                k_new = k_full[:, :, max_cache_len: max_cache_len + new_tokens, :]
                v_new = v_full[:, :, max_cache_len: max_cache_len + new_tokens, :]

                k = torch.cat([k_old, k_new], dim=2).clone()  # (1, H, new_len, D)
                v = torch.cat([v_old, v_new], dim=2).clone()

                new_cache.key_cache.append(k)
                new_cache.value_cache.append(v)

            PerRequestKVCacheManager._set_seen_tokens(new_cache, new_len)
            state.talker_kv_cache = new_cache
            state.cache_seq_length = new_len
