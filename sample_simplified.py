# ============================================================
# Stable Diffusion Cross-Attention Map 추출·분석 (정리본)
#
# 주요 기능:
#   - UNet cross attention map을 스텝별로 가로채 저장
#   - 토큰을 BOS / prompt / EOS+PAD 3그룹으로 분류
#   - CE, CosSim, Ratio, per-head HV CosSim 등 지표 계산
#   - Normal vs Memorized 프롬프트 비교 시각화
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import diffusers
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from PIL import Image


# ─────────────────────────────────────────────────────────────
# 1. AttentionStore
# ─────────────────────────────────────────────────────────────


class AttentionStore:
    """
    스텝별 Cross-Attention weight 저장소.
    step_attention[step_idx] = List[Tensor(heads, spatial, 77)]
    step_hv[step_idx]        = List[(av_special, av_prompt)]
      av_special / av_prompt : np.ndarray (n_heads, head_dim)
    """

    def __init__(self):
        self.step_attention: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.step_hv: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        self.step_counter: int = 0

    def reset(self):
        self.step_attention = defaultdict(list)
        self.step_hv = defaultdict(list)
        self.step_counter = 0


# ─────────────────────────────────────────────────────────────
# 2. 전역 변수 & get_attention_scores 패치
# ─────────────────────────────────────────────────────────────

_ACTIVE_STORE: Optional[AttentionStore] = None
_HV_SPECIAL_IDX: List[int] = []
_HV_PROMPT_IDX: List[int] = []


def _install_attention_patch():
    """
    Attention.get_attention_scores를 한 번만 패치.
    cross-attention(key seq_len == 77)일 때 attention weight를 캡쳐한다.
    """
    from diffusers.models.attention_processor import Attention as DiffAttn

    if getattr(DiffAttn.get_attention_scores, "_patched", False):
        return

    _orig = DiffAttn.get_attention_scores

    def _patched(self, query, key, attention_mask=None):
        attn_probs = _orig(self, query, key, attention_mask)

        global _ACTIVE_STORE
        if _ACTIVE_STORE is not None and key.shape[1] == 77:
            # CFG: 배치 앞 절반 = unconditional → 뒤 절반만 사용
            half = attn_probs.shape[0] // 2
            cap = attn_probs[half:] if half > 0 else attn_probs
            step = _ACTIVE_STORE.step_counter
            _ACTIVE_STORE.step_attention[step].append(cap.detach().cpu())

        return attn_probs

    _patched._patched = True
    DiffAttn.get_attention_scores = _patched
    print("get_attention_scores 패치 완료")


# ─────────────────────────────────────────────────────────────
# 3. _ValueAwareAttnProcessor
# ─────────────────────────────────────────────────────────────


class _ValueAwareAttnProcessor:
    """
    diffusers 표준 AttnProcessor와 동일한 forward 연산을 수행하되,
    텍스트 cross-attention(seq=77)에서 per-head weighted value sum을 캡쳐한다.
    """

    def __call__(
        self,
        attn,  # diffusers Attention 모듈 인스턴스 (Q/K/V 프로젝션, 헤드 수 등 보유)
        hidden_states: torch.Tensor,  # UNet 내부 특징맵 (batch, spatial, channels)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 텍스트 인코더 출력 (cross-attn 시 사용)
        attention_mask: Optional[torch.Tensor] = None,  # 어텐션 마스크 (보통 None)
        temb: Optional[torch.Tensor] = None,  # 시간 임베딩 (spatial_norm 사용 시)
        **kwargs,
    ) -> torch.Tensor:
        global _ACTIVE_STORE, _HV_SPECIAL_IDX, _HV_PROMPT_IDX

        # 잔차 연결(residual connection)을 위해 원본 입력 보관
        residual = hidden_states
        # encoder_hidden_states가 있으면 cross-attention, 없으면 self-attention
        is_cross = encoder_hidden_states is not None
        # 텍스트 cross-attention 여부: cross이면서 key 시퀀스 길이가 77(CLIP 토큰 수)인 경우
        is_txt_ca = is_cross and (encoder_hidden_states.shape[1] == 77)

        # spatial_norm이 있으면 시간 임베딩 기반 정규화 적용 (일부 모델에서 사용)
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 입력 차원 기억: 4D(B,C,H,W)이면 3D(B,H*W,C)로 변환
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        # query 시퀀스 길이 = 공간 해상도(H*W), key/value 시퀀스 길이 결정
        batch_size, seq_q, _ = hidden_states.shape
        seq_kv = (
            encoder_hidden_states.shape[1]  # cross-attn: 텍스트 토큰 수 (77)
            if encoder_hidden_states is not None
            else seq_q  # self-attn: 공간 해상도와 동일
        )

        # 어텐션 마스크가 있으면 multi-head 형태로 변환
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_kv, batch_size
            )

        # group_norm이 있으면 적용 (channels-first로 변환 후 다시 되돌림)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # Query 프로젝션: hidden_states → Q
        query = attn.to_q(hidden_states)

        # Key/Value 입력 결정: cross-attn이면 텍스트 인코더 출력, self-attn이면 hidden_states 자신
        _enc = encoder_hidden_states if is_cross else hidden_states
        # cross-attn에서 norm_cross 옵션이 켜져 있으면 인코더 출력 정규화
        if is_cross and getattr(attn, "norm_cross", False):
            _enc = attn.norm_encoder_hidden_states(_enc)

        # Key, Value 프로젝션
        key = attn.to_k(_enc)
        value = attn.to_v(_enc)

        # multi-head 처리: (B, seq, dim) → (B*n_heads, seq, head_dim) 으로 reshape
        query_b = attn.head_to_batch_dim(query)
        key_b = attn.head_to_batch_dim(key)
        value_b = attn.head_to_batch_dim(value)

        # attention score 계산: softmax(Q·K^T / sqrt(d)) → (B*n_heads, seq_q, seq_kv)
        # 이 내부에서 _install_attention_patch로 설치한 패치가 attention weight를 캡쳐함
        attn_probs = attn.get_attention_scores(query_b, key_b, attention_mask)

        # ── HV(Head-Value) 캡쳐: special/prompt 토큰별 weighted value sum 저장 ──
        if (
            _ACTIVE_STORE is not None  # 캡쳐 활성화 상태
            and is_txt_ca  # 텍스트 cross-attention일 때만
            and _HV_SPECIAL_IDX  # special 토큰 인덱스가 설정됨
            and _HV_PROMPT_IDX  # prompt 토큰 인덱스가 설정됨
        ):
            try:
                with torch.no_grad():  # 그래디언트 계산 불필요 (분석용)
                    BH, sq, sk = attn_probs.shape  # BH=batch*heads, sq=공간, sk=토큰(77)
                    n_heads = attn.heads  # 어텐션 헤드 수
                    B_act = BH // n_heads  # 실제 배치 크기 (CFG 포함)
                    head_dim = value_b.shape[-1]  # 헤드 당 차원 수

                    # CFG(Classifier-Free Guidance)에서 앞 절반은 unconditional → 뒤 절반(conditional)만 사용
                    half = B_act // 2
                    s_b = half * n_heads if half > 0 else 0  # conditional 시작 인덱스

                    # conditional 부분의 attention prob: (n_heads, sq, sk) — 배치 평균
                    ap = attn_probs[s_b:].reshape(-1, n_heads, sq, sk).mean(0).float()
                    # conditional 부분의 value: (n_heads, sk, head_dim) — 배치 평균
                    vb = (
                        value_b[s_b:].reshape(-1, n_heads, sk, head_dim).mean(0).float()
                    )

                    # 유효한 special/prompt 토큰 인덱스 필터링 (sk 범위 내)
                    s_idx = [i for i in _HV_SPECIAL_IDX if i < sk]
                    p_idx = [i for i in _HV_PROMPT_IDX if i < sk]

                    if s_idx and p_idx:
                        # special 토큰에 대한 attention prob과 value 추출
                        ap_s = ap[:, :, s_idx]  # (n_heads, sq, len(s_idx))
                        vb_s = vb[:, s_idx, :]  # (n_heads, len(s_idx), head_dim)
                        # prompt 토큰에 대한 attention prob과 value 추출
                        ap_p = ap[:, :, p_idx]  # (n_heads, sq, len(p_idx))
                        vb_p = vb[:, p_idx, :]  # (n_heads, len(p_idx), head_dim)

                        # special 토큰의 weighted value sum: sum_i(a_i * V_i)
                        # bmm: (n_heads, sq, len(s_idx)) @ (n_heads, len(s_idx), head_dim)
                        #       → (n_heads, sq, head_dim) → 공간 평균 → (n_heads, head_dim)
                        av_s = torch.bmm(
                            ap_s.reshape(n_heads, sq, len(s_idx)),
                            vb_s.reshape(n_heads, len(s_idx), head_dim),
                        ).mean(1)

                        # prompt 토큰의 weighted value sum: sum_j(a_j * V_j)
                        av_p = torch.bmm(
                            ap_p.reshape(n_heads, sq, len(p_idx)),
                            vb_p.reshape(n_heads, len(p_idx), head_dim),
                        ).mean(1)

                        # 현재 디퓨전 스텝의 HV 데이터로 저장
                        step = _ACTIVE_STORE.step_counter
                        _ACTIVE_STORE.step_hv[step].append(
                            (av_s.cpu().numpy(), av_p.cpu().numpy())
                        )
            except Exception:
                pass  # 캡쳐 실패 시 무시 (생성 흐름에 영향 주지 않음)

        # ── 표준 attention 출력 계산 (원래 AttnProcessor와 동일) ──
        # attention weight × value → weighted sum: (B*heads, sq, head_dim)
        hidden_states = torch.bmm(attn_probs, value_b)
        # multi-head를 다시 합침: (B*heads, sq, head_dim) → (B, sq, dim)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # 출력 선형 프로젝션
        hidden_states = attn.to_out[0](hidden_states)
        # 출력 드롭아웃
        hidden_states = attn.to_out[1](hidden_states)

        # 4D 입력이었으면 다시 4D로 복원
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        # 잔차 연결 (설정된 경우)
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        # 출력 스케일링 (설정된 경우, 기본값 1.0)
        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)

        return hidden_states


def _install_value_aware_processor(unet) -> None:
    """_ValueAwareAttnProcessor를 UNet에 설치한다."""
    try:
        unet.set_attn_processor(_ValueAwareAttnProcessor())
        print("ValueAwareAttnProcessor 설치 완료 (HV 캡쳐 활성)")
    except Exception as e:
        print(f"ValueAwareAttnProcessor 설치 실패, 표준 프로세서로 대체: {e}")
        try:
            from diffusers.models.attention_processor import AttnProcessor

            unet.set_attn_processor(AttnProcessor())
        except Exception:
            pass


def _set_hv_groups(groups: Dict[str, List[int]]) -> None:
    """generate()에서 프롬프트 토큰 그룹을 전역 변수에 반영."""
    global _HV_SPECIAL_IDX, _HV_PROMPT_IDX
    _HV_SPECIAL_IDX = groups["beginning"] + groups["summary"]
    _HV_PROMPT_IDX = groups["prompt"]


# ─────────────────────────────────────────────────────────────
# 4. 토큰 분류 유틸리티
# ─────────────────────────────────────────────────────────────


def classify_tokens(tokenizer, prompt: str) -> Dict[str, List[int]]:
    """
    CLIP 토크나이저 기준 토큰 인덱스를 3그룹으로 분류.
    beginning : <|startoftext|> (BOS)
    prompt    : 실제 단어 토큰
    summary   : <|endoftext|> + padding
    """
    BOS_ID = tokenizer.bos_token_id
    EOS_ID = tokenizer.eos_token_id

    enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids[0].tolist()

    groups: Dict[str, List[int]] = {"beginning": [], "prompt": [], "summary": []}
    found_eos = False
    for i, tid in enumerate(ids):
        if tid == BOS_ID:
            groups["beginning"].append(i)
        elif tid == EOS_ID or found_eos:
            groups["summary"].append(i)
            found_eos = True
        else:
            groups["prompt"].append(i)
    return groups


def get_token_labels(tokenizer, prompt: str) -> List[str]:
    enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids[0].tolist()
    raw = tokenizer.convert_ids_to_tokens(ids)
    return [
        t.replace("</w>", "")
        .replace("<|startoftext|>", "<BOS>")
        .replace("<|endoftext|>", "<EOS>")
        for t in raw
    ]


# ─────────────────────────────────────────────────────────────
# 5. 지표 계산 함수
# ─────────────────────────────────────────────────────────────


def aggregate_group_scores_per_step(
    store: AttentionStore,
    groups: Dict[str, List[int]],
    target_res: int = 16,
) -> Dict[str, np.ndarray]:
    """
    각 디퓨전 스텝 x 토큰 그룹별 attention score 합 반환.

    attn: (num_heads, H*W, 77)
      -> spatial 합산 -> (num_heads, 77)
      -> 헤드 평균 -> (77,)
      -> 전체 합 정규화 후 그룹별 합산
    """
    target_spatial = target_res * target_res
    steps = sorted(store.step_attention.keys())
    result: Dict[str, list] = {"beginning": [], "prompt": [], "summary": []}

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == target_spatial]

        if not maps:
            for g in result:
                result[g].append(np.nan)
            continue

        stacked = torch.stack(maps, 0).float().mean(0)
        per_tok = stacked.sum(dim=1).mean(dim=0).numpy()
        per_tok = per_tok / (per_tok.sum() + 1e-8)

        for g, indices in groups.items():
            valid = [i for i in indices if i < len(per_tok)]
            result[g].append(float(per_tok[valid].sum()) if valid else 0.0)

    return {g: np.array(v) for g, v in result.items()}


def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    """1-D 배열을 target_len으로 선형 보간 후 재정규화."""
    if len(arr) == target_len:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    interp = np.interp(x_new, x_old, arr)
    interp = np.clip(interp, 0, None)
    total = interp.sum()
    return interp / (total + 1e-12)


def _get_per_tok(maps: List[torch.Tensor], eps: float = 1e-12) -> np.ndarray:
    """attention map 리스트 -> 정규화된 (77,) 벡터."""
    per_tok = torch.stack(maps).float().mean(0).sum(1).mean(0).numpy()
    return per_tok / (per_tok.sum() + eps)


def compute_group_cross_entropy_per_step(
    store: AttentionStore,
    groups: Dict[str, List[int]],
    target_res: int = 16,
) -> Dict[str, np.ndarray]:
    """
    각 타임스텝 t에서 special / prompt token attention 분포 간 cross-entropy 계산.

    Returns:
        {
          "ce_s2p": np.ndarray (n_steps,),  CE(special -> prompt)
          "ce_p2s": np.ndarray (n_steps,),  CE(prompt  -> special)
          "h_s"   : np.ndarray (n_steps,),  H(special) self-entropy
          "h_p"   : np.ndarray (n_steps,),  H(prompt)  self-entropy
        }
    """
    EPS = 1e-12
    ts = target_res * target_res
    steps = sorted(store.step_attention.keys())

    special_idx = groups["beginning"] + groups["summary"]
    prompt_idx = groups["prompt"]

    ce_s2p_list, ce_p2s_list = [], []
    h_s_list, h_p_list = [], []

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == ts]
        if not maps:
            for lst in [ce_s2p_list, ce_p2s_list, h_s_list, h_p_list]:
                lst.append(np.nan)
            continue

        per_tok = _get_per_tok(maps, EPS)

        v_s = per_tok[[i for i in special_idx if i < len(per_tok)]]
        v_p = per_tok[[i for i in prompt_idx if i < len(per_tok)]]

        v_s = np.clip(v_s, 0, None)
        v_s = v_s / (v_s.sum() + EPS)
        v_p = np.clip(v_p, 0, None)
        v_p = v_p / (v_p.sum() + EPS)

        target_len = max(len(v_s), len(v_p))
        p_s = _interp_to_len(v_s, target_len)
        p_p = _interp_to_len(v_p, target_len)

        ce_s2p_list.append(float(-np.sum(p_s * np.log(p_p + EPS))))
        ce_p2s_list.append(float(-np.sum(p_p * np.log(p_s + EPS))))
        h_s_list.append(float(-np.sum(p_s * np.log(p_s + EPS))))
        h_p_list.append(float(-np.sum(p_p * np.log(p_p + EPS))))

    return {
        "ce_s2p": np.array(ce_s2p_list),
        "ce_p2s": np.array(ce_p2s_list),
        "h_s": np.array(h_s_list),
        "h_p": np.array(h_p_list),
    }


def compute_cosine_similarity_per_step(
    store: AttentionStore,
    groups: Dict[str, List[int]],
    target_res: int = 16,
) -> np.ndarray:
    """
    각 타임스텝에서 special(beginning+summary) / prompt token
    attention 벡터 간 코사인 유사도 계산.
    """
    EPS = 1e-12
    ts = target_res * target_res
    steps = sorted(store.step_attention.keys())

    special_idx = groups["beginning"] + groups["summary"]
    prompt_idx = groups["prompt"]

    cos_sim_list = []

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == ts]
        if not maps:
            cos_sim_list.append(np.nan)
            continue

        per_tok = _get_per_tok(maps, EPS)

        v_s = np.clip(per_tok[[i for i in special_idx if i < len(per_tok)]], 0, None)
        v_p = np.clip(per_tok[[i for i in prompt_idx if i < len(per_tok)]], 0, None)

        L = max(len(v_s), len(v_p), 1)
        v_s = _interp_to_len(v_s, L)
        v_p = _interp_to_len(v_p, L)

        dot = np.dot(v_s, v_p)
        norm = np.linalg.norm(v_s) * np.linalg.norm(v_p) + EPS
        cos_sim_list.append(float(dot / norm))

    return np.array(cos_sim_list)


def compute_summary_prompt_ratio_per_step(
    store: AttentionStore,
    groups: Dict[str, List[int]],
    target_res: int = 16,
) -> np.ndarray:
    """
    각 타임스텝 t에서: ratio(t) = sum A_t[I_summary] / sum A_t[I_prompt]
    summary = EOS + PAD (BOS 제외)
    """
    EPS = 1e-12
    ts = target_res * target_res
    steps = sorted(store.step_attention.keys())

    summary_idx = groups["summary"]
    prompt_idx = groups["prompt"]

    ratio_list = []

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == ts]
        if not maps:
            ratio_list.append(np.nan)
            continue

        per_tok = _get_per_tok(maps, EPS)

        sum_s = float(per_tok[[i for i in summary_idx if i < len(per_tok)]].sum())
        sum_p = float(per_tok[[i for i in prompt_idx if i < len(per_tok)]].sum())

        ratio_list.append(sum_s / (sum_p + EPS))

    return np.array(ratio_list)


def compute_head_av_cosine_per_step(
    store: AttentionStore,
    eps: float = 1e-12,
    n_heads_target: Optional[int] = None,
) -> np.ndarray:
    """
    Returns: (n_steps, n_heads) -- 스텝별 헤드별 cos_sim
    각 스텝에서 cross-attention layer들의 cos_sim을 평균냄.
    """
    steps = sorted(store.step_hv.keys())
    if not steps:
        return np.array([])

    if n_heads_target is None:
        cnt = Counter(
            av_s.shape[0]
            for step_list in store.step_hv.values()
            for (av_s, _) in step_list
        )
        n_heads_target = cnt.most_common(1)[0][0] if cnt else 8

    results = []
    for step in steps:
        layer_sims = []
        for av_s, av_p in store.step_hv[step]:
            if av_s.shape[0] != n_heads_target:
                continue
            H, d = av_s.shape
            cos = np.zeros(H)
            for h in range(H):
                dot = float(np.dot(av_s[h], av_p[h]))
                norm = float(np.linalg.norm(av_s[h]) * np.linalg.norm(av_p[h])) + eps
                cos[h] = dot / norm
            layer_sims.append(cos)

        if layer_sims:
            results.append(np.nanmean(layer_sims, axis=0))
        else:
            results.append(np.full(n_heads_target, np.nan))

    return np.array(results)


def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """1D 배열을 target length까지 nan으로 패딩."""
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.full(length - len(arr), np.nan)])


def _pad2d(arr: np.ndarray, target_steps: int) -> np.ndarray:
    """(n_steps, n_heads) 배열을 target_steps로 패딩 (nan 채움)."""
    s, h = arr.shape
    if s >= target_steps:
        return arr[:target_steps]
    pad = np.full((target_steps - s, h), np.nan)
    return np.concatenate([arr, pad], axis=0)


# ─────────────────────────────────────────────────────────────
# 6. 버전 감지
# ─────────────────────────────────────────────────────────────


def _diffusers_ver() -> tuple:
    v = diffusers.__version__.split(".")
    return (int(v[0]), int(v[1]))


# ─────────────────────────────────────────────────────────────
# 7. SDAttentionVisualizer
# ─────────────────────────────────────────────────────────────


class SDAttentionVisualizer:
    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"디바이스: {self.device}")
        print(f"diffusers: {diffusers.__version__}")

        from diffusers import DDIMScheduler, StableDiffusionPipeline

        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        print("모델 로딩 중...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(self.device)

        _install_value_aware_processor(self.pipe.unet)
        _install_attention_patch()

        self.store = AttentionStore()
        self.tokenizer = self.pipe.tokenizer

    # ── 이미지 생성 ───────────────────────────────────────────
    def generate(
        self, prompt: str, num_inference_steps: int = 20, seed: int = 42
    ) -> Image.Image:
        global _ACTIVE_STORE

        # HV 토큰 그룹 설정
        _grp = classify_tokens(self.tokenizer, prompt)
        _set_hv_groups(_grp)

        self.store.reset()
        _ACTIVE_STORE = self.store
        self.prompt = prompt
        self.tokens = get_token_labels(self.tokenizer, prompt)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        ver = _diffusers_ver()

        if ver >= (0, 22):

            def _cb(pipe, step_index, timestep, kwargs):
                self.store.step_counter += 1
                return kwargs

            print(f'생성 중 (new callback): "{prompt}"')
            result = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
                callback_on_step_end=_cb,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        else:

            def _cb(step, timestep, latents):
                self.store.step_counter += 1

            print(f'생성 중 (old callback): "{prompt}"')
            result = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
                callback=_cb,
                callback_steps=1,
            )

        _ACTIVE_STORE = None
        self.image = result.images[0]
        print(f"완료! 수집 스텝: {len(self.store.step_attention)}")
        return self.image

    # ── 내부 헬퍼 ────────────────────────────────────────────
    def _get_aggregated_attention(self, target_res: int = 16) -> torch.Tensor:
        ts = target_res * target_res
        collected = [
            a
            for maps in self.store.step_attention.values()
            for a in maps
            if a.shape[1] == ts
        ]
        if not collected:
            raise ValueError(f"해상도 {target_res}x{target_res} attention 없음")
        avg = torch.stack(collected).float().mean(0).mean(0)
        return avg.reshape(target_res, target_res, -1)

    # ─────────────────────────────────────────────────────────
    # 시각화 1: 토큰별 Attention Map
    # ─────────────────────────────────────────────────────────
    def visualize_per_token(
        self,
        target_res: int = 16,
        cmap: str = "hot",
        smooth: bool = True,
        save_path: Optional[str] = None,
    ):
        attn = self._get_aggregated_attention(target_res)
        seq_len = attn.shape[-1]
        tokens = self.tokens[:seq_len]
        n_tokens = len(tokens)

        ncols = min(n_tokens, 8)
        nrows = (n_tokens + ncols - 1) // ncols + 1

        fig = plt.figure(figsize=(ncols * 2.5, nrows * 2.5))
        fig.suptitle(
            f'Per-Token Cross-Attention Maps\n"{self.prompt}"',
            fontsize=12,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.3)

        ax_img = fig.add_subplot(gs[0, : ncols // 2])
        ax_img.imshow(self.image)
        ax_img.set_title("Generated Image")
        ax_img.axis("off")

        entropies = []
        for i in range(n_tokens):
            a = attn[:, :, i].numpy().flatten()
            a = a / (a.sum() + 1e-8)
            entropies.append(float(-np.sum(a * np.log(a + 1e-8))))

        ax_e = fig.add_subplot(gs[0, ncols // 2 :])
        bar_colors = [
            "red" if e < np.percentile(entropies, 25) else "steelblue"
            for e in entropies
        ]
        ax_e.bar(range(n_tokens), entropies, color=bar_colors, edgecolor="white")
        ax_e.set_xticks(range(n_tokens))
        ax_e.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax_e.set_title("Attention Entropy")
        ax_e.set_ylabel("Entropy")

        for idx in range(n_tokens):
            ax = fig.add_subplot(gs[idx // ncols + 1, idx % ncols])
            am = attn[:, :, idx].numpy()
            if smooth:
                am = (
                    np.array(
                        Image.fromarray((am / am.max() * 255).astype(np.uint8)).resize(
                            (256, 256), Image.BICUBIC
                        )
                    )
                    / 255.0
                )
            img_np = np.array(self.image.resize((256, 256))) / 255.0
            am_norm = am / (am.max() + 1e-8)
            ax.imshow(img_np, alpha=0.45)
            ax.imshow(am_norm, cmap=cmap, alpha=0.65, vmin=0, vmax=1)
            focused = entropies[idx] < np.percentile(entropies, 25)
            for sp in ax.spines.values():
                sp.set_edgecolor("red" if focused else "white")
                sp.set_linewidth(2.5 if focused else 1)
            ax.set_title(tokens[idx], fontsize=8, color="red" if focused else "black")
            ax.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.tight_layout()
            plt.show()

    # ─────────────────────────────────────────────────────────
    # 시각화 2: 스텝별 Entropy 변화
    # ─────────────────────────────────────────────────────────
    def visualize_entropy_over_steps(
        self, target_res: int = 16, save_path: Optional[str] = None
    ):
        ts = target_res * target_res
        step_ent = defaultdict(list)
        for step, maps in self.store.step_attention.items():
            for attn in maps:
                if attn.shape[1] != ts:
                    continue
                for si in range(attn.shape[2]):
                    a = attn.float().mean(0)[:, si].numpy()
                    a = a / (a.sum() + 1e-8)
                    step_ent[step].append(float(-np.sum(a * np.log(a + 1e-8))))

        steps = sorted(step_ent.keys())
        me = [np.mean(step_ent[s]) for s in steps]
        se = [np.std(step_ent[s]) for s in steps]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, me, color="steelblue", lw=2, marker="o", ms=4)
        ax.fill_between(
            steps,
            np.array(me) - np.array(se),
            np.array(me) + np.array(se),
            alpha=0.25,
            color="steelblue",
        )
        ax.set(
            xlabel="Diffusion Step",
            ylabel="Attention Entropy",
            title=f'Entropy over Steps\n"{self.prompt}"',
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150) if save_path else plt.show()

    # ─────────────────────────────────────────────────────────
    # 시각화 3: 스텝별 Token Group Attention Score 합
    # ─────────────────────────────────────────────────────────
    def visualize_token_group_attention_over_steps(
        self,
        target_res: int = 16,
        save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        4개 서브플롯:
          1 라인 플롯  -- 3그룹 attention 비율 추이
          2 스택 영역  -- 100% 누적 비율
          3 히트맵     -- 토큰 위치 x 스텝 2D 맵
          4 이미지 + 토큰 분류 요약표
        """
        groups = classify_tokens(self.tokenizer, self.prompt)
        labels = get_token_labels(self.tokenizer, self.prompt)
        scores = aggregate_group_scores_per_step(self.store, groups, target_res)
        steps = np.arange(len(scores["beginning"]))
        step_labels = [f"T-{i}" for i in steps]

        COLORS = {"beginning": "#E74C3C", "prompt": "#2ECC71", "summary": "#3498DB"}
        LABELS = {
            "beginning": "Beginning <BOS>",
            "prompt": "Prompt Tokens (words)",
            "summary": "Summary <EOS>/<PAD>",
        }

        fig = plt.figure(figsize=(18, 22), facecolor="#0F1117")
        fig.suptitle(
            f'Token Group Attention Analysis\n"{self.prompt}"',
            fontsize=14,
            fontweight="bold",
            color="white",
            y=0.98,
        )
        gs = gridspec.GridSpec(
            4,
            2,
            figure=fig,
            hspace=0.55,
            wspace=0.35,
            height_ratios=[1.2, 1.2, 1.5, 1.0],
        )

        ax_line = fig.add_subplot(gs[0, :])
        ax_stack = fig.add_subplot(gs[1, :])
        ax_heat = fig.add_subplot(gs[2, :])
        ax_img = fig.add_subplot(gs[3, 0])
        ax_table = fig.add_subplot(gs[3, 1])

        dark = "#1A1D2E"
        for ax in [ax_line, ax_stack, ax_heat, ax_img, ax_table]:
            ax.set_facecolor(dark)
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_edgecolor("#444466")

        xtick_step = max(1, len(steps) // 10)

        # 1 라인 플롯
        for g in ["beginning", "prompt", "summary"]:
            y = scores[g]
            ax_line.plot(
                steps, y, color=COLORS[g], lw=2.5, marker="o", ms=4, label=LABELS[g]
            )
            ax_line.fill_between(steps, y, alpha=0.12, color=COLORS[g])

        s_max_idx = int(np.nanargmax(scores["summary"]))
        ax_line.axvline(s_max_idx, color=COLORS["summary"], ls="--", alpha=0.6, lw=1.5)
        ax_line.annotate(
            f"Summary max\n(Step {s_max_idx})",
            xy=(s_max_idx, scores["summary"][s_max_idx]),
            xytext=(
                s_max_idx + max(1, len(steps) // 12),
                scores["summary"][s_max_idx] * 1.15,
            ),
            color=COLORS["summary"],
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=COLORS["summary"], lw=1.5),
        )
        ax_line.set_title(
            "Step-wise Attention Score Sum per Token Group",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax_line.set_xlabel("Diffusion Step", color="#AAAACC", fontsize=10)
        ax_line.set_ylabel("Normalized Attention Sum", color="#AAAACC", fontsize=10)
        ax_line.set_xticks(steps[::xtick_step])
        ax_line.set_xticklabels(
            step_labels[::xtick_step],
            rotation=30,
            ha="right",
            fontsize=8,
            color="white",
        )
        ax_line.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax_line.legend(
            fontsize=9,
            loc="upper right",
            facecolor=dark,
            labelcolor="white",
            edgecolor="#444466",
        )
        ax_line.grid(True, alpha=0.2, color="#555577")

        # 2 스택 영역
        tot = scores["beginning"] + scores["prompt"] + scores["summary"] + 1e-8
        ax_stack.stackplot(
            steps,
            scores["beginning"] / tot,
            scores["prompt"] / tot,
            scores["summary"] / tot,
            labels=[LABELS["beginning"], LABELS["prompt"], LABELS["summary"]],
            colors=[COLORS["beginning"], COLORS["prompt"], COLORS["summary"]],
            alpha=0.82,
        )
        ax_stack.set_title(
            "Stacked Area: Relative Proportion of Each Token Group",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax_stack.set_xlabel("Diffusion Step", color="#AAAACC", fontsize=10)
        ax_stack.set_ylabel("Proportion", color="#AAAACC", fontsize=10)
        ax_stack.set_xlim(steps[0], steps[-1])
        ax_stack.set_ylim(0, 1)
        ax_stack.set_xticks(steps[::xtick_step])
        ax_stack.set_xticklabels(
            step_labels[::xtick_step],
            rotation=30,
            ha="right",
            fontsize=8,
            color="white",
        )
        ax_stack.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_stack.legend(
            fontsize=9,
            loc="upper right",
            facecolor=dark,
            labelcolor="white",
            edgecolor="#444466",
        )
        ax_stack.grid(True, alpha=0.15, color="#555577", axis="y")

        # 3 히트맵
        ts = target_res * target_res
        sorted_steps = sorted(self.store.step_attention.keys())
        seq_len = self.tokenizer.model_max_length

        heat = np.zeros((seq_len, len(sorted_steps)))
        for ci, step in enumerate(sorted_steps):
            maps = [a for a in self.store.step_attention[step] if a.shape[1] == ts]
            if not maps:
                continue
            pt = torch.stack(maps).float().mean(0).sum(1).mean(0).numpy()
            pt = pt / (pt.sum() + 1e-8)
            heat[:, ci] = pt[:seq_len]

        vmax = np.percentile(heat[heat > 0], 98) if heat.max() > 0 else 1
        im = ax_heat.imshow(
            heat, aspect="auto", cmap="hot", interpolation="nearest", vmin=0, vmax=vmax
        )
        cb = plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
        cb.set_label("Normalized Attention", color="white")
        cb.ax.yaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")

        for g, color in COLORS.items():
            idx = groups[g]
            if not idx:
                continue
            lo, hi = min(idx) - 0.5, max(idx) + 0.5
            ax_heat.axhline(lo, color=color, lw=1.2, ls="--", alpha=0.8)
            ax_heat.axhline(hi, color=color, lw=1.2, ls="--", alpha=0.8)
            ax_heat.text(
                len(sorted_steps) + 0.3,
                (lo + hi) / 2,
                g[:3].upper(),
                color=color,
                fontsize=7,
                va="center",
                fontweight="bold",
            )

        ytick_pos = list(range(0, seq_len, max(1, seq_len // 15)))
        ax_heat.set_yticks(ytick_pos)
        ax_heat.set_yticklabels(
            [labels[i] for i in ytick_pos], fontsize=7, color="white"
        )
        ax_heat.set_xticks(range(0, len(sorted_steps), xtick_step))
        ax_heat.set_xticklabels(
            [f"T-{i}" for i in range(0, len(sorted_steps), xtick_step)],
            rotation=30,
            ha="right",
            fontsize=8,
            color="white",
        )
        ax_heat.set_title(
            "Per-Token Attention Heatmap over Steps", color="white", fontsize=12, pad=10
        )
        ax_heat.set_xlabel("Diffusion Step", color="#AAAACC", fontsize=10)
        ax_heat.set_ylabel("Token Position", color="#AAAACC", fontsize=10)

        # 4-A 원본 이미지
        ax_img.imshow(self.image)
        ax_img.set_title("Generated Image", color="white", fontsize=11)
        ax_img.axis("off")

        # 4-B 토큰 분류 요약표
        ax_table.axis("off")
        ax_table.set_title("Token Classification Summary", color="white", fontsize=11)

        table_data, col_hdrs = (
            [],
            ["Group", "#Tokens", "Mean Attn", "Final Attn", "Tokens"],
        )
        for g in ["beginning", "prompt", "summary"]:
            idx = groups[g]
            preview = ", ".join(labels[i] for i in idx[:5])
            if len(idx) > 5:
                preview += f" (+{len(idx) - 5})"
            table_data.append(
                [
                    g.capitalize(),
                    str(len(idx)),
                    f"{np.nanmean(scores[g]):.4f}",
                    f"{scores[g][-1] if len(scores[g]) > 0 else 0:.4f}",
                    preview,
                ]
            )

        tbl = ax_table.table(
            cellText=table_data,
            colLabels=col_hdrs,
            cellLoc="center",
            loc="center",
            bbox=[0, 0.2, 1, 0.7],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        g_order = ["beginning", "prompt", "summary"]
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor(dark)
            cell.set_edgecolor("#444466")
            cell.set_text_props(color="white")
            if row == 0:
                cell.set_facecolor("#2C2F45")
                cell.set_text_props(color="white", fontweight="bold")
            elif 1 <= row <= 3:
                r, gg, b, _ = mcolors.to_rgba(COLORS[g_order[row - 1]])
                cell.set_facecolor((r * 0.3, gg * 0.3, b * 0.3, 1.0))

        if save_path:
            plt.savefig(
                save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
            )
            print(f"저장: {save_path}")
        else:
            plt.tight_layout()
            plt.show()

        return scores

    # ─────────────────────────────────────────────────────────
    # 리스트 기반 비교 (HV 포함)
    # ─────────────────────────────────────────────────────────
    def compare_prompt_lists_hv(
        self,
        normal_prompts: List[str],
        mem_prompts: List[str],
        num_inference_steps: int = 20,
        seed: int = 42,
        target_res: int = 16,
        save_path: Optional[str] = None,
    ):
        """
        Normal vs Memorized 프롬프트 리스트를 비교.
        CE / CosSim / Ratio / per-head HV CosSim 시각화 모두 포함.
        """
        print(f"\n{'=' * 60}")
        print(f"  Normal    prompts: {len(normal_prompts)}개")
        print(f"  Memorized prompts: {len(mem_prompts)}개")
        print(f"{'=' * 60}")

        print("\n[Phase 1] Normal prompts 처리 중...")
        nm = compute_metrics_for_prompt_list(
            self,
            normal_prompts,
            num_inference_steps=num_inference_steps,
            seed=seed,
            target_res=target_res,
            group_label="Normal",
        )

        print("\n[Phase 2] Memorized prompts 처리 중...")
        mm = compute_metrics_for_prompt_list(
            self,
            mem_prompts,
            num_inference_steps=num_inference_steps,
            seed=seed,
            target_res=target_res,
            group_label="Memorized",
        )

        # CE / CosSim / ratio 그래프
        visualize_list_comparison(
            nm,
            mm,
            normal_label="Normal Prompts",
            mem_label="Memorized Prompts",
            save_path=save_path,
        )
        ratio_save = save_path.replace(".png", "_ratio.png") if save_path else None
        visualize_summary_prompt_ratio(
            nm,
            mm,
            normal_label="Normal Prompts",
            mem_label="Memorized Prompts",
            save_path=ratio_save,
            show_individual=True,
        )

        # per-head HV CosSim 그래프
        hv_save = save_path.replace(".png", "_hv.png") if save_path else None
        if (
            nm.get("hv_cos_mean") is not None
            and nm["hv_cos_mean"].size > 0
            and mm.get("hv_cos_mean") is not None
            and mm["hv_cos_mean"].size > 0
        ):
            visualize_head_av_cosine(
                cos_normal=nm["hv_cos_mean"],
                cos_mem=mm["hv_cos_mean"],
                normal_label="Normal Prompts",
                mem_label="Memorized Prompts",
                save_path=hv_save,
            )
        else:
            print("HV cosine 데이터 없음 -- _ValueAwareAttnProcessor가 설치됐는지 확인")

        # 콘솔 요약
        print("\n" + "=" * 70)
        print(f"{'지표':<22} {'Normal':>18} {'Memorized':>18} {'d(M-N)':>10}")
        print("=" * 70)
        for key, label in [
            ("ce_s2p_mean", "CE(s->p)"),
            ("ce_p2s_mean", "CE(p->s)"),
            ("h_s_mean", "H(special)"),
            ("h_p_mean", "H(prompt)"),
            ("diff_mean", "dCE"),
            ("cos_sim_mean", "CosSim(attn)"),
            ("ratio_mean", "Ratio(EOS/P)"),
        ]:
            n_val = np.nanmean(nm[key])
            m_val = np.nanmean(mm[key])
            print(f"{label:<22} {n_val:>18.4f} {m_val:>18.4f} {m_val - n_val:>+10.4f}")

        if nm.get("hv_cos_mean") is not None and nm["hv_cos_mean"].size > 0:
            print("-" * 70)
            print("  HV CosSim  (mean over steps, per head):")
            for h in range(nm["hv_cos_mean"].shape[1]):
                nv = np.nanmean(nm["hv_cos_mean"][:, h])
                mv = np.nanmean(mm["hv_cos_mean"][:, h])
                marker = (
                    " < d max"
                    if abs(mv - nv)
                    == max(
                        abs(
                            np.nanmean(mm["hv_cos_mean"][:, hh])
                            - np.nanmean(nm["hv_cos_mean"][:, hh])
                        )
                        for hh in range(nm["hv_cos_mean"].shape[1])
                    )
                    else ""
                )
                print(
                    f"    Head {h:2d}:  Normal={nv:+.4f}  Mem={mv:+.4f}"
                    f"  d={mv - nv:+.4f}{marker}"
                )
        print("=" * 70)

        return nm, mm


# ─────────────────────────────────────────────────────────────
# 8. 시각화 함수
# ─────────────────────────────────────────────────────────────


def visualize_list_comparison(
    normal_metrics: Dict,
    mem_metrics: Dict,
    normal_label: str = "Normal Prompts",
    mem_label: str = "Memorized Prompts",
    save_path: Optional[str] = None,
):
    """
    compute_metrics_for_prompt_list() 결과 두 개를 받아
    5개 패널 + 요약 테이블을 그린다.
    """
    nm = normal_metrics
    mm = mem_metrics

    sn = np.arange(len(nm["ce_s2p_mean"]))
    sm = np.arange(len(mm["ce_s2p_mean"]))

    C_NOR = "#4A90D9"
    C_MEM = "#E74C3C"

    fig = plt.figure(figsize=(20, 22))
    gs = gridspec.GridSpec(
        5, 3, figure=fig, hspace=0.55, wspace=0.35, height_ratios=[1, 1, 1, 1, 1]
    )

    def _plot_band(ax, steps, mean, std, color, label, ls="-"):
        ax.plot(steps, mean, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.18, color=color)

    # 1 CE(special -> prompt)
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_band(
        ax1,
        sn,
        nm["ce_s2p_mean"],
        nm["ce_s2p_std"],
        C_NOR,
        f"{normal_label} (n={nm['n_prompts']})",
    )
    _plot_band(
        ax1,
        sm,
        mm["ce_s2p_mean"],
        mm["ce_s2p_std"],
        C_MEM,
        f"{mem_label} (n={mm['n_prompts']})",
    )
    ax1.set_title(
        "CE(special -> prompt)  [mean +/- std]", fontsize=10, fontweight="bold"
    )
    ax1.set_xlabel("Diffusion Step")
    ax1.set_ylabel("Cross-Entropy (nats)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2 CE(prompt -> special)
    ax2 = fig.add_subplot(gs[1, :2])
    _plot_band(ax2, sn, nm["ce_p2s_mean"], nm["ce_p2s_std"], C_NOR, f"{normal_label}")
    _plot_band(ax2, sm, mm["ce_p2s_mean"], mm["ce_p2s_std"], C_MEM, f"{mem_label}")
    ax2.set_title(
        "CE(prompt -> special)  [mean +/- std]", fontsize=10, fontweight="bold"
    )
    ax2.set_xlabel("Diffusion Step")
    ax2.set_ylabel("Cross-Entropy (nats)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # 3 Self-Entropy
    ax3 = fig.add_subplot(gs[2, :2])
    _plot_band(
        ax3,
        sn,
        nm["h_s_mean"],
        nm["h_s_std"],
        C_NOR,
        f"{normal_label} H(special)",
        ls="-",
    )
    _plot_band(
        ax3,
        sn,
        nm["h_p_mean"],
        nm["h_p_std"],
        C_NOR,
        f"{normal_label} H(prompt)",
        ls="--",
    )
    _plot_band(
        ax3, sm, mm["h_s_mean"], mm["h_s_std"], C_MEM, f"{mem_label} H(special)", ls="-"
    )
    _plot_band(
        ax3, sm, mm["h_p_mean"], mm["h_p_std"], C_MEM, f"{mem_label} H(prompt)", ls="--"
    )
    ax3.set_title(
        "Self-Entropy H(special) / H(prompt)  [mean +/- std]",
        fontsize=10,
        fontweight="bold",
    )
    ax3.set_xlabel("Diffusion Step")
    ax3.set_ylabel("Entropy (nats)")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(alpha=0.3)

    # 4 Cosine Similarity
    ax4 = fig.add_subplot(gs[3, :2])
    _plot_band(ax4, sn, nm["cos_sim_mean"], nm["cos_sim_std"], C_NOR, f"{normal_label}")
    _plot_band(ax4, sm, mm["cos_sim_mean"], mm["cos_sim_std"], C_MEM, f"{mem_label}")
    ax4.axhline(0, color="k", lw=0.8, ls="--")
    ax4.set_title(
        "Cosine Similarity(special, prompt)  [mean +/- std]",
        fontsize=10,
        fontweight="bold",
    )
    ax4.set_xlabel("Diffusion Step")
    ax4.set_ylabel("Cosine Similarity")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # 5 dCE = CE(s->p) - CE(p->s)
    ax5 = fig.add_subplot(gs[4, :2])
    _plot_band(ax5, sn, nm["diff_mean"], nm["diff_std"], C_NOR, f"{normal_label} dCE")
    _plot_band(ax5, sm, mm["diff_mean"], mm["diff_std"], C_MEM, f"{mem_label} dCE")
    ax5.axhline(0, color="k", lw=0.8, ls="--")
    ax5.fill_between(
        sn, nm["diff_mean"], 0, where=nm["diff_mean"] < 0, alpha=0.12, color=C_NOR
    )
    ax5.fill_between(
        sm, mm["diff_mean"], 0, where=mm["diff_mean"] < 0, alpha=0.12, color=C_MEM
    )
    ax5.set_title(
        "dCE = CE(special->prompt) - CE(prompt->special)  [mean +/- std]",
        fontsize=10,
        fontweight="bold",
    )
    ax5.set_xlabel("Diffusion Step")
    ax5.set_ylabel("dCE (nats)")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # 우측 요약 테이블
    ax_tbl = fig.add_subplot(gs[:, 2])
    ax_tbl.axis("off")

    rows = [
        ["지표", normal_label[:18], mem_label[:18], "차이(M-N)"],
        ["n_prompts", str(nm["n_prompts"]), str(mm["n_prompts"]), "-"],
        [
            "Mean CE(s->p)",
            f"{np.nanmean(nm['ce_s2p_mean']):.4f}",
            f"{np.nanmean(mm['ce_s2p_mean']):.4f}",
            f"{np.nanmean(mm['ce_s2p_mean']) - np.nanmean(nm['ce_s2p_mean']):+.4f}",
        ],
        [
            "Mean CE(p->s)",
            f"{np.nanmean(nm['ce_p2s_mean']):.4f}",
            f"{np.nanmean(mm['ce_p2s_mean']):.4f}",
            f"{np.nanmean(mm['ce_p2s_mean']) - np.nanmean(nm['ce_p2s_mean']):+.4f}",
        ],
        [
            "Mean H(special)",
            f"{np.nanmean(nm['h_s_mean']):.4f}",
            f"{np.nanmean(mm['h_s_mean']):.4f}",
            f"{np.nanmean(mm['h_s_mean']) - np.nanmean(nm['h_s_mean']):+.4f}",
        ],
        [
            "Mean H(prompt)",
            f"{np.nanmean(nm['h_p_mean']):.4f}",
            f"{np.nanmean(mm['h_p_mean']):.4f}",
            f"{np.nanmean(mm['h_p_mean']) - np.nanmean(nm['h_p_mean']):+.4f}",
        ],
        [
            "Mean dCE",
            f"{np.nanmean(nm['diff_mean']):.4f}",
            f"{np.nanmean(mm['diff_mean']):.4f}",
            f"{np.nanmean(mm['diff_mean']) - np.nanmean(nm['diff_mean']):+.4f}",
        ],
        [
            "Mean CosSim",
            f"{np.nanmean(nm['cos_sim_mean']):.4f}",
            f"{np.nanmean(mm['cos_sim_mean']):.4f}",
            f"{np.nanmean(mm['cos_sim_mean']) - np.nanmean(nm['cos_sim_mean']):+.4f}",
        ],
        [
            "Min  CosSim",
            f"{np.nanmin(nm['cos_sim_mean']):.4f}",
            f"{np.nanmin(mm['cos_sim_mean']):.4f}",
            f"{np.nanmin(mm['cos_sim_mean']) - np.nanmin(nm['cos_sim_mean']):+.4f}",
        ],
    ]

    tbl = ax_tbl.table(
        cellText=[r[1:] for r in rows[1:]],
        rowLabels=[r[0] for r in rows[1:]],
        colLabels=rows[0][1:],
        loc="center",
        cellLoc="center",
        bbox=[0, 0.05, 1, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:
            try:
                val = float(cell.get_text().get_text())
                cell.set_facecolor("#FADBD8" if val > 0 else "#D6EAF8")
            except ValueError:
                pass

    ax_tbl.set_title(
        "요약 테이블\n(M-N = Mem minus Normal)", fontsize=10, fontweight="bold", pad=8
    )

    ax_tbl.text(
        0.0,
        0.0,
        f"Normal prompts ({nm['n_prompts']}):\n"
        + "\n".join(
            f"  - {p[:55]}{'...' if len(p) > 55 else ''}"
            for p in [r["prompt"] for r in nm["per_prompt"]]
        )
        + f"\n\nMem prompts ({mm['n_prompts']}):\n"
        + "\n".join(
            f"  - {p[:55]}{'...' if len(p) > 55 else ''}"
            for p in [r["prompt"] for r in mm["per_prompt"]]
        ),
        transform=ax_tbl.transAxes,
        fontsize=6.5,
        va="top",
        color="#555555",
        bbox=dict(
            facecolor="#F8F9FA",
            alpha=0.7,
            edgecolor="#CCCCCC",
            boxstyle="round,pad=0.3",
        ),
    )

    fig.suptitle(
        f"Normal vs Memorized Prompts -- Cross-Attention Metrics per Timestep\n"
        f"(Normal n={nm['n_prompts']}  |  Memorized n={mm['n_prompts']}  |  "
        f"shaded = +/-1 std)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return nm, mm


def visualize_summary_prompt_ratio(
    normal_metrics: Dict,
    mem_metrics: Dict,
    normal_label: str = "Normal Prompts",
    mem_label: str = "Memorized Prompts",
    save_path: Optional[str] = None,
    show_individual: bool = True,
):
    """
    ratio(t) = sum A_t[EOS/PAD] / sum A_t[prompt words]
    타임스텝별 Normal vs Memorized 평균 비율 비교 그래프.
    """
    nm = normal_metrics
    mm = mem_metrics

    sn = np.arange(len(nm["ratio_mean"]))
    sm = np.arange(len(mm["ratio_mean"]))

    C_NOR = "#4A90D9"
    C_MEM = "#E74C3C"
    C_DIFF = "#F39C12"

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(
        3 if show_individual else 2,
        3,
        figure=fig,
        hspace=0.50,
        wspace=0.30,
        height_ratios=[2, 1, 1.5] if show_individual else [2, 1],
    )

    # 1 메인: ratio mean +/- std
    ax1 = fig.add_subplot(gs[0, :2])

    ax1.plot(
        sn,
        nm["ratio_mean"],
        color=C_NOR,
        lw=2.5,
        label=f"{normal_label} (n={nm['n_prompts']})",
    )
    ax1.fill_between(
        sn,
        nm["ratio_mean"] - nm["ratio_std"],
        nm["ratio_mean"] + nm["ratio_std"],
        alpha=0.20,
        color=C_NOR,
    )

    ax1.plot(
        sm,
        mm["ratio_mean"],
        color=C_MEM,
        lw=2.5,
        label=f"{mem_label} (n={mm['n_prompts']})",
    )
    ax1.fill_between(
        sm,
        mm["ratio_mean"] - mm["ratio_std"],
        mm["ratio_mean"] + mm["ratio_std"],
        alpha=0.20,
        color=C_MEM,
    )

    ax1.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.7, label="ratio = 1.0")

    for steps_arr, mean_arr, color, tag in [
        (sn, nm["ratio_mean"], C_NOR, normal_label[:12]),
        (sm, mm["ratio_mean"], C_MEM, mem_label[:12]),
    ]:
        valid = ~np.isnan(mean_arr)
        if valid.any():
            peak_idx = steps_arr[valid][np.argmax(mean_arr[valid])]
            peak_val = mean_arr[valid].max()
            ax1.annotate(
                f"{tag}\npeak={peak_val:.3f}",
                xy=(peak_idx, peak_val),
                xytext=(peak_idx + max(1, len(steps_arr) // 10), peak_val * 1.05),
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

    ax1.set_title(
        "Summary(EOS/PAD) / Prompt Token Attention Ratio  [mean +/- std]",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Diffusion Step")
    ax1.set_ylabel("ratio  =  sum A[EOS/PAD] / sum A[prompt]")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2 차이: Mem - Normal
    ax2 = fig.add_subplot(gs[1, :2])

    common_len = min(len(nm["ratio_mean"]), len(mm["ratio_mean"]))
    delta = mm["ratio_mean"][:common_len] - nm["ratio_mean"][:common_len]
    steps_c = np.arange(common_len)

    ax2.plot(steps_c, delta, color=C_DIFF, lw=2, label="Mem - Normal")
    ax2.fill_between(
        steps_c,
        delta,
        0,
        where=delta > 0,
        alpha=0.25,
        color=C_MEM,
        label="Mem > Normal",
    )
    ax2.fill_between(
        steps_c,
        delta,
        0,
        where=delta <= 0,
        alpha=0.25,
        color=C_NOR,
        label="Normal >= Mem",
    )
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_title("d_ratio = Mem - Normal", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Diffusion Step")
    ax2.set_ylabel("d_ratio")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3 개별 프롬프트 선
    if show_individual:
        ax3 = fig.add_subplot(gs[2, :2])

        for row in nm["ratio_per_prompt"]:
            steps_i = np.arange(len(row))
            ax3.plot(steps_i, row, color=C_NOR, lw=0.8, alpha=0.35)
        ax3.plot(
            sn, nm["ratio_mean"], color=C_NOR, lw=2.5, label=f"{normal_label} mean"
        )

        for row in mm["ratio_per_prompt"]:
            steps_i = np.arange(len(row))
            ax3.plot(steps_i, row, color=C_MEM, lw=0.8, alpha=0.35)
        ax3.plot(sm, mm["ratio_mean"], color=C_MEM, lw=2.5, label=f"{mem_label} mean")

        ax3.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.6)
        ax3.set_title("개별 프롬프트 ratio 곡선", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Diffusion Step")
        ax3.set_ylabel("ratio")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

    # 우측: 요약 수치 테이블
    ax_tbl = fig.add_subplot(gs[:, 2])
    ax_tbl.axis("off")

    n_ratio = nm["ratio_mean"]
    m_ratio = mm["ratio_mean"]

    rows = [
        ["지표", normal_label[:16], mem_label[:16], "d(M-N)"],
        ["n_prompts", str(nm["n_prompts"]), str(mm["n_prompts"]), "-"],
        [
            "Mean ratio",
            f"{np.nanmean(n_ratio):.4f}",
            f"{np.nanmean(m_ratio):.4f}",
            f"{np.nanmean(m_ratio) - np.nanmean(n_ratio):+.4f}",
        ],
        [
            "Max  ratio",
            f"{np.nanmax(n_ratio):.4f}",
            f"{np.nanmax(m_ratio):.4f}",
            f"{np.nanmax(m_ratio) - np.nanmax(n_ratio):+.4f}",
        ],
        [
            "Min  ratio",
            f"{np.nanmin(n_ratio):.4f}",
            f"{np.nanmin(m_ratio):.4f}",
            f"{np.nanmin(m_ratio) - np.nanmin(n_ratio):+.4f}",
        ],
        ["Std  ratio", f"{np.nanstd(n_ratio):.4f}", f"{np.nanstd(m_ratio):.4f}", "-"],
        [
            "Step@peak ratio",
            str(int(np.nanargmax(n_ratio))),
            str(int(np.nanargmax(m_ratio))),
            "-",
        ],
        ["Mean d_ratio (M-N)", "-", "-", f"{np.nanmean(delta):+.4f}"],
    ]

    tbl = ax_tbl.table(
        cellText=[r[1:] for r in rows[1:]],
        rowLabels=[r[0] for r in rows[1:]],
        colLabels=rows[0][1:],
        loc="upper center",
        cellLoc="center",
        bbox=[0, 0.45, 1, 0.50],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.9)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:
            try:
                val = float(cell.get_text().get_text().replace("\u2212", "-"))
                cell.set_facecolor("#FADBD8" if val > 0 else "#D6EAF8")
            except ValueError:
                pass

    ax_tbl.set_title(
        "수치 요약\n(ratio = EOS/PAD / prompt)\nBOS 제외",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )

    ax_tbl.text(
        0.0,
        0.40,
        f"Normal ({nm['n_prompts']}):\n"
        + "\n".join(
            f"  {p['prompt'][:50]}{'...' if len(p['prompt']) > 50 else ''}"
            for p in nm["per_prompt"]
        )
        + f"\n\nMem ({mm['n_prompts']}):\n"
        + "\n".join(
            f"  {p['prompt'][:50]}{'...' if len(p['prompt']) > 50 else ''}"
            for p in mm["per_prompt"]
        ),
        transform=ax_tbl.transAxes,
        fontsize=6,
        va="top",
        color="#444444",
        bbox=dict(
            facecolor="#F4F6F7",
            alpha=0.8,
            edgecolor="#CCCCCC",
            boxstyle="round,pad=0.3",
        ),
    )

    fig.suptitle(
        "Summary(EOS/PAD) vs Prompt Token Attention Ratio\n"
        f"Normal n={nm['n_prompts']}  |  Memorized n={mm['n_prompts']}  |  "
        "shaded = +/-1 std  |  BOS 제외",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_head_av_cosine(
    cos_normal: np.ndarray,
    cos_mem: np.ndarray,
    normal_label: str = "Normal Prompts",
    mem_label: str = "Memorized Prompts",
    save_path: Optional[str] = None,
):
    """
    5-panel 시각화:
    1/2 Normal / Memorized step x head 히트맵
    3   Difference 히트맵 (Mem - Normal)
    4   헤드 평균, 스텝별 선 그래프
    5   스텝 평균, 헤드별 막대 그래프
    """
    n_steps_n, n_heads = cos_normal.shape
    n_steps_m = cos_mem.shape[0]
    C_NOR = "#4A90D9"
    C_MEM = "#E74C3C"

    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.50, wspace=0.32, height_ratios=[1.2, 1.4, 1.0]
    )

    def _heatmap(ax, data, title, cmap="RdYlGn", vmin=-1, vmax=1):
        im = ax.imshow(
            data.T,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Cosine Similarity")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Diffusion Step", fontsize=9)
        ax.set_ylabel("Head Index", fontsize=9)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
        step_arr = np.arange(data.shape[0])
        xt = step_arr[:: max(1, len(step_arr) // 8)]
        ax.set_xticks(xt)
        ax.set_xticklabels(xt, fontsize=7)

    # 1 Normal 히트맵
    ax1 = fig.add_subplot(gs[0, 0])
    _heatmap(ax1, cos_normal, f"{normal_label}\nHV CosSim per head")

    # 2 Memorized 히트맵
    ax2 = fig.add_subplot(gs[0, 1])
    _heatmap(ax2, cos_mem, f"{mem_label}\nHV CosSim per head")

    # 3 차이 히트맵 (Mem - Normal)
    ax3 = fig.add_subplot(gs[1, :])
    common = min(n_steps_n, n_steps_m)
    diff = cos_mem[:common] - cos_normal[:common]
    vabs = np.nanmax(np.abs(diff)) + 1e-6
    im3 = ax3.imshow(
        diff.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vabs,
        vmax=vabs,
        interpolation="nearest",
    )
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.015, pad=0.01)
    cb3.set_label("d CosSim  (Mem - Normal)")
    ax3.set_title("d CosSim = Mem - Normal", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Diffusion Step", fontsize=9)
    ax3.set_ylabel("Head Index", fontsize=9)
    ax3.set_yticks(range(n_heads))
    ax3.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
    xt3 = np.arange(common)[:: max(1, common // 10)]
    ax3.set_xticks(xt3)
    ax3.set_xticklabels(xt3, fontsize=7)

    # 4 헤드 평균 per step
    ax4 = fig.add_subplot(gs[2, 0])
    sn = np.arange(n_steps_n)
    sm = np.arange(n_steps_m)
    mn = np.nanmean(cos_normal, axis=1)
    stn = np.nanstd(cos_normal, axis=1)
    mm_mean = np.nanmean(cos_mem, axis=1)
    stm = np.nanstd(cos_mem, axis=1)

    ax4.plot(
        sn, mn, color=C_NOR, lw=2, label=f"{normal_label}  (mean+/-std over heads)"
    )
    ax4.fill_between(sn, mn - stn, mn + stn, alpha=0.18, color=C_NOR)
    ax4.plot(sm, mm_mean, color=C_MEM, lw=2, label=f"{mem_label}")
    ax4.fill_between(sm, mm_mean - stm, mm_mean + stm, alpha=0.18, color=C_MEM)
    ax4.axhline(0, color="k", lw=0.8, ls="--")

    for steps_a, mean_a, color, tag in [
        (sn, mn, C_NOR, "N"),
        (sm, mm_mean, C_MEM, "M"),
    ]:
        valid = ~np.isnan(mean_a)
        if valid.any():
            pk = steps_a[valid][np.argmax(mean_a[valid])]
            ax4.annotate(
                f"{tag} peak\n{mean_a[valid].max():.3f}",
                xy=(pk, mean_a[pk]),
                xytext=(pk + max(1, len(steps_a) // 10), mean_a[pk] + 0.04),
                fontsize=7,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax4.set_title(
        "헤드 평균 CosSim per step  [mean +/- std]", fontsize=10, fontweight="bold"
    )
    ax4.set_xlabel("Diffusion Step")
    ax4.set_ylabel("Cosine Similarity")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.15, 1.05)

    # 5 스텝 평균 per head -- 막대 그래프
    ax5 = fig.add_subplot(gs[2, 1])
    mphn = np.nanmean(cos_normal, axis=0)
    mphm = np.nanmean(cos_mem, axis=0)
    errn = np.nanstd(cos_normal, axis=0)
    errm = np.nanstd(cos_mem, axis=0)
    x = np.arange(n_heads)
    w = 0.38
    ax5.bar(
        x - w / 2,
        mphn,
        width=w,
        color=C_NOR,
        alpha=0.85,
        yerr=errn,
        capsize=3,
        label=f"{normal_label}",
    )
    ax5.bar(
        x + w / 2,
        mphm,
        width=w,
        color=C_MEM,
        alpha=0.85,
        yerr=errm,
        capsize=3,
        label=f"{mem_label}",
    )
    ax5.axhline(0, color="k", lw=0.8)

    delta_per_head = mphm - mphn
    top_head = int(np.nanargmax(np.abs(delta_per_head)))
    ax5.axvspan(
        top_head - 0.5,
        top_head + 0.5,
        alpha=0.12,
        color="gold",
        label=f"H{top_head}: d={delta_per_head[top_head]:+.3f}",
    )

    ax5.set_title(
        "스텝 평균 CosSim per head (mean +/- std over steps)",
        fontsize=10,
        fontweight="bold",
    )
    ax5.set_xlabel("Head Index")
    ax5.set_ylabel("Mean Cosine Similarity")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Per-Head  CosSim(  sum_i_special a_i*V_i ,  sum_j_prompt a_j*V_j  )\n"
        "[averaged over all cross-attn layers]",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 9. compute_metrics_for_prompt_list (HV 포함 통합 버전)
# ─────────────────────────────────────────────────────────────


def compute_metrics_for_prompt_list(
    viz,
    prompt_list: List[str],
    num_inference_steps: int = 20,
    seed: int = 42,
    target_res: int = 16,
    group_label: str = "group",
) -> Dict:
    """
    프롬프트 리스트의 각 프롬프트에 대해 CE / CosSim / Ratio / HV CosSim 계산 후
    스텝별 mean +/- std 반환.

    Returns:
        {
          "ce_s2p_mean", "ce_s2p_std",
          "ce_p2s_mean", "ce_p2s_std",
          "h_s_mean",    "h_s_std",
          "h_p_mean",    "h_p_std",
          "cos_sim_mean","cos_sim_std",
          "diff_mean",   "diff_std",
          "ratio_mean",  "ratio_std",  "ratio_per_prompt",
          "hv_cos_mean", "hv_cos_std",
          "n_prompts": int,
          "per_prompt": List[Dict],
        }
    """
    all_ce_s2p, all_ce_p2s = [], []
    all_h_s, all_h_p = [], []
    all_cos_sim = []
    all_ratio = []
    all_hv_cos = []
    per_prompt_records = []

    total = len(prompt_list)
    for i, prompt in enumerate(prompt_list):
        print(f'\n  [{group_label}] {i + 1}/{total}: "{prompt[:60]}"')
        viz.generate(prompt, num_inference_steps=num_inference_steps, seed=seed)

        store = deepcopy(viz.store)
        groups = classify_tokens(viz.tokenizer, prompt)

        m = compute_group_cross_entropy_per_step(store, groups, target_res)
        cs = compute_cosine_similarity_per_step(store, groups, target_res)
        ratio = compute_summary_prompt_ratio_per_step(store, groups, target_res)
        hv = compute_head_av_cosine_per_step(store)

        all_ce_s2p.append(m["ce_s2p"])
        all_ce_p2s.append(m["ce_p2s"])
        all_h_s.append(m["h_s"])
        all_h_p.append(m["h_p"])
        all_cos_sim.append(cs)
        all_ratio.append(ratio)
        all_hv_cos.append(hv)

        per_prompt_records.append(
            {
                "prompt": prompt,
                "ce_s2p": m["ce_s2p"],
                "ce_p2s": m["ce_p2s"],
                "h_s": m["h_s"],
                "h_p": m["h_p"],
                "cos_sim": cs,
                "ratio": ratio,
                "hv_cos": hv,
            }
        )

    max_steps = max(len(x) for x in all_ce_s2p)

    def _stack1d(lst):
        return np.stack([_pad_to(x, max_steps) for x in lst])

    stk_s2p = _stack1d(all_ce_s2p)
    stk_p2s = _stack1d(all_ce_p2s)
    stk_hs = _stack1d(all_h_s)
    stk_hp = _stack1d(all_h_p)
    stk_cs = _stack1d(all_cos_sim)
    stk_dif = stk_s2p - stk_p2s
    stk_ratio = _stack1d(all_ratio)

    n_heads_hv = max(
        (x.shape[1] for x in all_hv_cos if x.ndim == 2 and x.size > 0),
        default=8,
    )
    stk_hv = np.stack(
        [
            _pad2d(x, max_steps)
            if (x.ndim == 2 and x.size > 0)
            else np.full((max_steps, n_heads_hv), np.nan)
            for x in all_hv_cos
        ]
    )

    return {
        "ce_s2p_mean": np.nanmean(stk_s2p, axis=0),
        "ce_s2p_std": np.nanstd(stk_s2p, axis=0),
        "ce_p2s_mean": np.nanmean(stk_p2s, axis=0),
        "ce_p2s_std": np.nanstd(stk_p2s, axis=0),
        "h_s_mean": np.nanmean(stk_hs, axis=0),
        "h_s_std": np.nanstd(stk_hs, axis=0),
        "h_p_mean": np.nanmean(stk_hp, axis=0),
        "h_p_std": np.nanstd(stk_hp, axis=0),
        "cos_sim_mean": np.nanmean(stk_cs, axis=0),
        "cos_sim_std": np.nanstd(stk_cs, axis=0),
        "diff_mean": np.nanmean(stk_dif, axis=0),
        "diff_std": np.nanstd(stk_dif, axis=0),
        "ratio_mean": np.nanmean(stk_ratio, axis=0),
        "ratio_std": np.nanstd(stk_ratio, axis=0),
        "ratio_per_prompt": stk_ratio,
        "hv_cos_mean": np.nanmean(stk_hv, axis=0),
        "hv_cos_std": np.nanstd(stk_hv, axis=0),
        "n_prompts": total,
        "per_prompt": per_prompt_records,
    }


# ─────────────────────────────────────────────────────────────
# 10. __main__
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    NORMAL_PROMPTS = [
        "a photo of an astronaut riding a horse on the moon",
        "a painting of a cat sitting on a red couch",
        "a beautiful sunset over the ocean",
        "a dog playing in the park on a sunny day",
    ]

    MEM_PROMPTS = [
        "The No Limits Business Woman Podcast",
        "Mothers influence on her young hippo",
        "The Happy Scientist",
        "Foyer painted in HABANERO",
    ]

    viz = SDAttentionVisualizer(device="cuda")

    nm, mm = viz.compare_prompt_lists_hv(
        normal_prompts=NORMAL_PROMPTS,
        mem_prompts=MEM_PROMPTS,
        num_inference_steps=50,
        seed=42,
        target_res=16,
        save_path="normal_vs_mem_comparison.png",
    )
