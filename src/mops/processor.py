"""Cross-attention 통계 수집 AttnProcessor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mops.stats_store import CrossAttentionStatsStore

CLIP_MAX_TOKENS = 77


def _spatial_stats(tensor: torch.Tensor) -> tuple[float, float, float, float]:
    """(num_spatial,) 텐서로부터 mean, var, max, min을 반환한다."""
    return (
        tensor.mean().item(),
        tensor.var(correction=0).item(),
        tensor.max().item(),
        tensor.min().item(),
    )


class StatsAttnProcessor:
    """
    diffusers 표준 AttnProcessor와 동일한 forward 연산을 수행하되,
    텍스트 cross-attention(seq=77)에서 per-head 통계를 즉시 계산하여 저장한다.

    수집 항목:
    - alpha_prompt / alpha_summary: attention score 부분합 (spatial별 통계)
    - cossim_value: spatial 평균 후 prompt vs summary weighted value의 cosine similarity (head당 스칼라)
    - cossim_ov: spatial 평균 후 W_o 적용(OV circuit) cosine similarity (head당 스칼라)
    """

    def __init__(self, stats_store: CrossAttentionStatsStore, layer_name: str) -> None:
        self.stats_store = stats_store
        self.layer_name = layer_name
        # W_o weight의 float32 캐시 (첫 호출 시 초기화)
        self._cached_output_weight: torch.Tensor | None = None

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        is_cross_attention = encoder_hidden_states is not None
        is_text_cross_attention = is_cross_attention and (encoder_hidden_states.shape[1] == CLIP_MAX_TOKENS)

        # spatial_norm 처리 (일부 모델에서 사용)
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 4D 입력이면 3D로 변환
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size_4d, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size_4d, channels, height * width).transpose(1, 2)

        batch_size, sequence_length_query, _ = hidden_states.shape
        sequence_length_kv = (
            encoder_hidden_states.shape[1] if encoder_hidden_states is not None else sequence_length_query
        )

        # 어텐션 마스크 준비
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length_kv, batch_size)

        # group_norm 처리
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Q/K/V 프로젝션
        query = attn.to_q(hidden_states)
        encoder_input = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and getattr(attn, "norm_cross", False):
            encoder_input = attn.norm_encoder_hidden_states(encoder_input)
        key = attn.to_k(encoder_input)
        value = attn.to_v(encoder_input)

        # multi-head reshape: (batch, seq, dim) → (batch*heads, seq, head_dim)
        query_batched = attn.head_to_batch_dim(query)
        key_batched = attn.head_to_batch_dim(key)
        value_batched = attn.head_to_batch_dim(value)

        # attention score 계산: softmax(QK^T / sqrt(d))
        attention_probs = attn.get_attention_scores(query_batched, key_batched, attention_mask)

        # 텍스트 cross-attention일 때만 통계 수집
        if is_text_cross_attention and self.stats_store.prompt_token_indices:
            with torch.no_grad():
                self._compute_and_store_stats(attn, attention_probs, value_batched)

        # 표준 attention 출력 계산
        hidden_states = torch.bmm(attention_probs, value_batched)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # 4D 복원
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size_4d, channels, height, width)

        # 잔차 연결
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states

    def _compute_and_store_stats(
        self,
        attn,
        attention_probs: torch.Tensor,
        value_batched: torch.Tensor,
    ) -> None:
        """
        attention score와 value로부터 per-head 통계를 벡터화하여 계산 후 store에 기록.

        attention_probs: (batch*heads, num_spatial, 77)
        value_batched:   (batch*heads, 77, head_dim)
        """
        total_batch_heads, num_spatial, num_text_tokens = attention_probs.shape
        num_heads = attn.heads
        actual_batch_size = total_batch_heads // num_heads
        head_dim = value_batched.shape[-1]

        # CFG: conditional half만 사용 (뒤쪽 절반)
        conditional_batch_size = actual_batch_size // 2
        conditional_start_index = conditional_batch_size * num_heads if conditional_batch_size > 0 else 0

        # conditional 부분 추출 후 batch 평균 → (num_heads, num_spatial, 77)
        effective_batch = conditional_batch_size if conditional_batch_size > 0 else actual_batch_size
        attention_probs_cond = (
            attention_probs[conditional_start_index:]
            .reshape(effective_batch, num_heads, num_spatial, num_text_tokens)
            .mean(0)
            .float()
        )
        # (num_heads, 77, head_dim)
        value_vectors_cond = (
            value_batched[conditional_start_index:]
            .reshape(effective_batch, num_heads, num_text_tokens, head_dim)
            .mean(0)
            .float()
        )

        # 토큰 인덱스를 텐서로 변환 (GPU에서 인덱싱 효율)
        device = attention_probs_cond.device
        prompt_idx = torch.as_tensor(self.stats_store.prompt_token_indices, device=device)
        summary_idx = torch.as_tensor(self.stats_store.summary_token_indices, device=device)

        # === 1) alpha: attention score 부분합 (벡터화) ===
        # (num_heads, num_spatial)
        alpha_prompt_all_heads = attention_probs_cond[:, :, prompt_idx].sum(dim=-1)
        alpha_summary_all_heads = attention_probs_cond[:, :, summary_idx].sum(dim=-1)

        # alpha_ratio: spatial 각 위치에서 prompt/summary 비율 → spatial 평균 (num_heads,)
        EPS = 1e-12
        alpha_ratio_per_head = (alpha_prompt_all_heads / (alpha_summary_all_heads + EPS)).mean(dim=1)

        # === 2) value weighted sum → spatial 평균 후 cosine similarity ===
        # (num_heads, num_spatial, len_p) @ (num_heads, len_p, head_dim) → (num_heads, num_spatial, head_dim)
        weighted_value_prompt = torch.bmm(
            attention_probs_cond[:, :, prompt_idx],
            value_vectors_cond[:, prompt_idx, :],
        )
        weighted_value_summary = torch.bmm(
            attention_probs_cond[:, :, summary_idx],
            value_vectors_cond[:, summary_idx, :],
        )
        # spatial 먼저 평균 → (num_heads, head_dim)
        wv_prompt_mean = weighted_value_prompt.mean(dim=1)
        wv_summary_mean = weighted_value_summary.mean(dim=1)
        # cosine similarity: (num_heads,)
        cossim_value_per_head = F.cosine_similarity(wv_prompt_mean, wv_summary_mean, dim=-1)

        # === 3) W_o 적용 (OV circuit) — spatial 평균 후 cosine similarity ===
        if self._cached_output_weight is None:
            self._cached_output_weight = attn.to_out[0].weight.float()
        output_weight = self._cached_output_weight  # (out_dim, inner_dim)

        # head별 W_o slice: (num_heads, head_dim, out_dim)
        output_weight_per_head = output_weight.T.reshape(num_heads, head_dim, -1)

        # (num_heads, head_dim) @ (num_heads, head_dim, out_dim) → (num_heads, out_dim)
        # unsqueeze(1)로 bmm 호환 → squeeze(1)로 복원
        ov_prompt = torch.bmm(wv_prompt_mean.unsqueeze(1), output_weight_per_head).squeeze(1)
        ov_summary = torch.bmm(wv_summary_mean.unsqueeze(1), output_weight_per_head).squeeze(1)
        # cosine similarity: (num_heads,)
        cossim_ov_per_head = F.cosine_similarity(ov_prompt, ov_summary, dim=-1)

        # === 4) per-head 통계를 CPU로 한꺼번에 전송 후 기록 ===
        # alpha: (num_heads, num_spatial) → per-head (mean, var, max, min)
        # alpha_ratio, cossim: (num_heads,) → per-head 스칼라
        alpha_metrics = torch.stack([alpha_prompt_all_heads, alpha_summary_all_heads])  # (2, num_heads, num_spatial)
        scalar_metrics = torch.stack([alpha_ratio_per_head, cossim_value_per_head, cossim_ov_per_head]).cpu()

        for head_index in range(num_heads):
            ap_mean, ap_var, ap_max, ap_min = _spatial_stats(alpha_metrics[0, head_index])
            as_mean, as_var, as_max, as_min = _spatial_stats(alpha_metrics[1, head_index])

            self.stats_store.add_record(
                self.layer_name,
                head_index,
                alpha_prompt_mean=ap_mean,
                alpha_prompt_var=ap_var,
                alpha_prompt_max=ap_max,
                alpha_prompt_min=ap_min,
                alpha_summary_mean=as_mean,
                alpha_summary_var=as_var,
                alpha_summary_max=as_max,
                alpha_summary_min=as_min,
                alpha_ratio=scalar_metrics[0, head_index].item(),
                cossim_value=scalar_metrics[1, head_index].item(),
                cossim_ov=scalar_metrics[2, head_index].item(),
            )
