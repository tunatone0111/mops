# ============================================================
# Stable Diffusion Cross-Attention Map 시각화 (수정 완료판)
# 핵심 변경: Attention.forward 교체 → get_attention_scores 패치
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import diffusers
from collections import defaultdict
from PIL import Image
from typing import List, Dict, Optional
from copy import deepcopy

# ─────────────────────────────────────────────────────────────
# 1. AttentionStore
# ─────────────────────────────────────────────────────────────

class AttentionStore:
    """
    스텝별 Cross-Attention weight 저장소.
    step_attention[step_idx] = List[Tensor(heads, spatial, 77)]
    """
    def __init__(self):
        self.step_attention: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.step_counter: int = 0

    def reset(self):
        self.step_attention = defaultdict(list)
        self.step_counter = 0


# ─────────────────────────────────────────────────────────────
# 2. ✅ 핵심 수정: get_attention_scores 패치
#    - forward 전체 교체 X → score 계산 직후만 가로채기
#    - 버전 무관하게 동작
#    - cross-attention 판별: key.shape[1] == 77 (CLIP 토큰 수)
# ─────────────────────────────────────────────────────────────

_ACTIVE_STORE: Optional[AttentionStore] = None

def _install_attention_patch():
    """
    Attention.get_attention_scores를 한 번만 패치.
    diffusers AttnProcessor2_0(Flash Attention)을 우회하기 위해
    표준 AttnProcessor로 강제 전환 후 적용.
    """
    from diffusers.models.attention_processor import Attention as DiffAttn

    # 이미 패치됐으면 재설치하지 않음
    if getattr(DiffAttn.get_attention_scores, '_patched', False):
        return

    _orig = DiffAttn.get_attention_scores

    def _patched(self, query, key, attention_mask=None):
        attn_probs = _orig(self, query, key, attention_mask)

        global _ACTIVE_STORE
        # cross-attention 판별: key seq_len == 77 (CLIP max_length)
        if _ACTIVE_STORE is not None and key.shape[1] == 77:
            # CFG 사용 시 배치 앞 절반 = unconditional → 뒤 절반만 사용
            half = attn_probs.shape[0] // 2
            cap  = attn_probs[half:] if half > 0 else attn_probs
            step = _ACTIVE_STORE.step_counter
            _ACTIVE_STORE.step_attention[step].append(cap.detach().cpu())

        return attn_probs

    _patched._patched = True
    DiffAttn.get_attention_scores = _patched
    print("🔧 get_attention_scores 패치 완료 (forward 무수정)")


def _force_standard_attn_processor(unet):
    """
    Flash Attention(AttnProcessor2_0)은 get_attention_scores를 우회하므로
    표준 AttnProcessor로 강제 교체.
    """
    try:
        from diffusers.models.attention_processor import AttnProcessor
        unet.set_attn_processor(AttnProcessor())
        print("🔄 표준 AttnProcessor로 교체 완료 (Flash Attention 비활성화)")
    except Exception as e:
        print(f"⚠️ AttnProcessor 교체 실패 (무시 가능): {e}")


# ─────────────────────────────────────────────────────────────
# 3. 토큰 분류 유틸리티
# ─────────────────────────────────────────────────────────────

def classify_tokens(tokenizer, prompt: str) -> Dict[str, List[int]]:
    """
    CLIP 토크나이저 기준 토큰 인덱스를 3그룹으로 분류.
    beginning : <|startoftext|>  (BOS)
    prompt    : 실제 단어 토큰
    summary   : <|endoftext|> + padding
    """
    BOS_ID = tokenizer.bos_token_id  # 49406
    EOS_ID = tokenizer.eos_token_id  # 49407

    enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids[0].tolist()

    groups = {"beginning": [], "prompt": [], "summary": []}
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
    ids  = enc.input_ids[0].tolist()
    raw  = tokenizer.convert_ids_to_tokens(ids)
    return [t.replace("</w>", "")
             .replace("<|startoftext|>", "<BOS>")
             .replace("<|endoftext|>", "<EOS>")
            for t in raw]


# ─────────────────────────────────────────────────────────────
# 4. 스텝별 Token Group Attention 집계
# ─────────────────────────────────────────────────────────────

def aggregate_group_scores_per_step(
    store: AttentionStore,
    groups: Dict[str, List[int]],
    target_res: int = 16,
) -> Dict[str, np.ndarray]:
    """
    각 디퓨전 스텝 × 토큰 그룹별 attention score 합 반환.

    attn: (num_heads, H*W, 77)
      → spatial 합산 → (num_heads, 77)
      → 헤드 평균 → (77,)
      → 전체 합 정규화 후 그룹별 합산
    """
    target_spatial = target_res * target_res
    steps  = sorted(store.step_attention.keys())
    result = {"beginning": [], "prompt": [], "summary": []}

    for step in steps:
        maps = [a for a in store.step_attention[step]
                if a.shape[1] == target_spatial]

        if not maps:
            for g in result:
                result[g].append(np.nan)
            continue

        # 레이어 평균 → (heads, spatial, 77)
        stacked  = torch.stack(maps, 0).float().mean(0)
        # spatial 합산 → (heads, 77)
        per_tok  = stacked.sum(dim=1).mean(dim=0).numpy()  # (77,)
        per_tok  = per_tok / (per_tok.sum() + 1e-8)

        for g, indices in groups.items():
            valid = [i for i in indices if i < len(per_tok)]
            result[g].append(float(per_tok[valid].sum()) if valid else 0.0)

    for g in result:
        result[g] = np.array(result[g])
    return result


# ─────────────────────────────────────────────────────────────
# 5. 버전 감지 유틸리티
# ─────────────────────────────────────────────────────────────

def _diffusers_ver() -> tuple:
    v = diffusers.__version__.split(".")
    return (int(v[0]), int(v[1]))


# ─────────────────────────────────────────────────────────────
# 6. SDAttentionVisualizer
# ─────────────────────────────────────────────────────────────

class SDAttentionVisualizer:

    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4",
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  디바이스: {self.device}")
        print(f"📦 diffusers: {diffusers.__version__}")

        from diffusers import StableDiffusionPipeline, DDIMScheduler

        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        print("📥 모델 로딩 중...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(self.device)

        # ✅ Flash Attention → 표준 AttnProcessor 강제 교체
        _force_standard_attn_processor(self.pipe.unet)

        # ✅ get_attention_scores 패치 (forward 무수정)
        _install_attention_patch()

        self.store     = AttentionStore()
        self.tokenizer = self.pipe.tokenizer

    # ── 이미지 생성 ───────────────────────────────────────────
    def generate(self, prompt: str,
                 num_inference_steps: int = 20,
                 seed: int = 42) -> Image.Image:

        global _ACTIVE_STORE
        self.store.reset()
        _ACTIVE_STORE = self.store        # 전역 참조 연결
        self.prompt   = prompt
        self.tokens   = get_token_labels(self.tokenizer, prompt)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        ver = _diffusers_ver()

        # ── 버전별 callback ──────────────────────────────────
        if ver >= (0, 22):
            def _cb(pipe, step_index, timestep, kwargs):
                self.store.step_counter += 1
                return kwargs

            print(f"🎨 생성 중 (new callback): \"{prompt}\"")
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

            print(f"🎨 생성 중 (old callback): \"{prompt}\"")
            result = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
                callback=_cb,
                callback_steps=1,
            )

        _ACTIVE_STORE = None              # 참조 해제
        self.image = result.images[0]
        print(f"✅ 완료! 수집 스텝: {len(self.store.step_attention)}")
        return self.image

    # ── 내부 헬퍼 ────────────────────────────────────────────
    def _get_aggregated_attention(self, target_res: int = 16) -> torch.Tensor:
        ts = target_res * target_res
        collected = [a for maps in self.store.step_attention.values()
                       for a in maps if a.shape[1] == ts]
        if not collected:
            raise ValueError(f"해상도 {target_res}x{target_res} attention 없음")
        avg = torch.stack(collected).float().mean(0).mean(0)  # (spatial, 77)
        return avg.reshape(target_res, target_res, -1)

    # ─────────────────────────────────────────────────────────
    # 시각화 1: 토큰별 Attention Map
    # ─────────────────────────────────────────────────────────
    def visualize_per_token(self, target_res: int = 16,
                             cmap: str = "hot", smooth: bool = True,
                             save_path: Optional[str] = None):
        attn     = self._get_aggregated_attention(target_res)
        seq_len  = attn.shape[-1]
        tokens   = self.tokens[:seq_len]
        n_tokens = len(tokens)

        ncols = min(n_tokens, 8)
        nrows = (n_tokens + ncols - 1) // ncols + 1

        fig = plt.figure(figsize=(ncols * 2.5, nrows * 2.5))
        fig.suptitle(f'Per-Token Cross-Attention Maps\n"{self.prompt}"',
                     fontsize=12, fontweight="bold")
        gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.5, wspace=0.3)

        ax_img = fig.add_subplot(gs[0, :ncols//2])
        ax_img.imshow(self.image); ax_img.set_title("Generated Image"); ax_img.axis("off")

        entropies = []
        for i in range(n_tokens):
            a = attn[:, :, i].numpy().flatten()
            a = a / (a.sum() + 1e-8)
            entropies.append(float(-np.sum(a * np.log(a + 1e-8))))

        ax_e = fig.add_subplot(gs[0, ncols//2:])
        bar_colors = ["red" if e < np.percentile(entropies, 25) else "steelblue"
                      for e in entropies]
        ax_e.bar(range(n_tokens), entropies, color=bar_colors, edgecolor="white")
        ax_e.set_xticks(range(n_tokens))
        ax_e.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax_e.set_title("Attention Entropy (낮을수록 집중)"); ax_e.set_ylabel("Entropy")

        for idx in range(n_tokens):
            ax = fig.add_subplot(gs[idx // ncols + 1, idx % ncols])
            am = attn[:, :, idx].numpy()
            if smooth:
                am = np.array(Image.fromarray(
                    (am / am.max() * 255).astype(np.uint8)
                ).resize((256, 256), Image.BICUBIC)) / 255.0
            img_np  = np.array(self.image.resize((256, 256))) / 255.0
            am_norm = am / (am.max() + 1e-8)
            ax.imshow(img_np, alpha=0.45)
            ax.imshow(am_norm, cmap=cmap, alpha=0.65, vmin=0, vmax=1)
            focused = entropies[idx] < np.percentile(entropies, 25)
            for sp in ax.spines.values():
                sp.set_edgecolor("red" if focused else "white"); sp.set_linewidth(2.5 if focused else 1)
            ax.set_title(tokens[idx] + ("\n⚠️" if focused else ""),
                          fontsize=8, color="red" if focused else "black"); ax.axis("off")

        (plt.savefig(save_path, dpi=150, bbox_inches="tight")
         if save_path else (plt.tight_layout(), plt.show()))

    # ─────────────────────────────────────────────────────────
    # 시각화 2: 스텝별 Entropy 변화
    # ─────────────────────────────────────────────────────────
    def visualize_entropy_over_steps(self, target_res: int = 16,
                                      save_path: Optional[str] = None):
        ts = target_res * target_res
        step_ent = defaultdict(list)
        for step, maps in self.store.step_attention.items():
            for attn in maps:
                if attn.shape[1] != ts: continue
                for si in range(attn.shape[2]):
                    a = attn.float().mean(0)[:, si].numpy()
                    a = a / (a.sum() + 1e-8)
                    step_ent[step].append(float(-np.sum(a * np.log(a + 1e-8))))

        steps = sorted(step_ent.keys())
        me = [np.mean(step_ent[s]) for s in steps]
        se = [np.std(step_ent[s])  for s in steps]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, me, color="steelblue", lw=2, marker="o", ms=4)
        ax.fill_between(steps, np.array(me)-np.array(se),
                                np.array(me)+np.array(se), alpha=0.25, color="steelblue")
        ax.set(xlabel="Diffusion Step", ylabel="Attention Entropy",
               title=f'Entropy over Steps\n"{self.prompt}"')
        ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(save_path, dpi=150) if save_path else plt.show()

    # ─────────────────────────────────────────────────────────
    # ✅ 시각화 3: 스텝별 Token Group Attention Score 합
    # ─────────────────────────────────────────────────────────
    def visualize_token_group_attention_over_steps(
        self, target_res: int = 16, save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        4개 서브플롯:
          ① 라인 플롯  — 3그룹 attention 비율 추이
          ② 스택 영역  — 100% 누적 비율
          ③ 히트맵     — 토큰 위치 × 스텝 2D 맵
          ④ 이미지 + 토큰 분류 요약표
        """
        groups = classify_tokens(self.tokenizer, self.prompt)
        labels = get_token_labels(self.tokenizer, self.prompt)
        scores = aggregate_group_scores_per_step(self.store, groups, target_res)
        steps  = np.arange(len(scores["beginning"]))
        step_labels = [f"T-{i}" for i in steps]

        COLORS = {"beginning": "#E74C3C", "prompt": "#2ECC71", "summary": "#3498DB"}
        LABELS = {"beginning": "Beginning <BOS>",
                  "prompt":    "Prompt Tokens (words)",
                  "summary":   "Summary <EOS>/<PAD>"}

        # ── 레이아웃 ───────────────────────────────────────
        fig = plt.figure(figsize=(18, 22), facecolor="#0F1117")
        fig.suptitle(f'Token Group Attention Analysis\n"{self.prompt}"',
                     fontsize=14, fontweight="bold", color="white", y=0.98)
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                               height_ratios=[1.2, 1.2, 1.5, 1.0])

        ax_line  = fig.add_subplot(gs[0, :])
        ax_stack = fig.add_subplot(gs[1, :])
        ax_heat  = fig.add_subplot(gs[2, :])
        ax_img   = fig.add_subplot(gs[3, 0])
        ax_table = fig.add_subplot(gs[3, 1])

        dark = "#1A1D2E"
        for ax in [ax_line, ax_stack, ax_heat, ax_img, ax_table]:
            ax.set_facecolor(dark)
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#444466")

        xtick_step = max(1, len(steps) // 10)

        # ════════════════════════════════════════════════
        # ① 라인 플롯
        # ════════════════════════════════════════════════
        for g in ["beginning", "prompt", "summary"]:
            y = scores[g]
            ax_line.plot(steps, y, color=COLORS[g], lw=2.5,
                         marker="o", ms=4, label=LABELS[g])
            ax_line.fill_between(steps, y, alpha=0.12, color=COLORS[g])

        # Summary 최대 지점 주석
        s_max_idx = int(np.nanargmax(scores["summary"]))
        ax_line.axvline(s_max_idx, color=COLORS["summary"],
                        ls="--", alpha=0.6, lw=1.5)
        ax_line.annotate(
            f"Summary 최대\n(Step {s_max_idx})",
            xy=(s_max_idx, scores["summary"][s_max_idx]),
            xytext=(s_max_idx + max(1, len(steps)//12),
                    scores["summary"][s_max_idx] * 1.15),
            color=COLORS["summary"], fontsize=8,
            arrowprops=dict(arrowstyle="->", color=COLORS["summary"], lw=1.5),
        )
        ax_line.set_title("① Step-wise Attention Score Sum per Token Group",
                           color="white", fontsize=12, pad=10)
        ax_line.set_xlabel("Diffusion Step  (왼쪽=노이즈 T  →  오른쪽=완성)",
                            color="#AAAACC", fontsize=10)
        ax_line.set_ylabel("Normalized Attention Sum", color="#AAAACC", fontsize=10)
        ax_line.set_xticks(steps[::xtick_step])
        ax_line.set_xticklabels(step_labels[::xtick_step],
                                 rotation=30, ha="right", fontsize=8, color="white")
        ax_line.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax_line.legend(fontsize=9, loc="upper right",
                       facecolor=dark, labelcolor="white", edgecolor="#444466")
        ax_line.grid(True, alpha=0.2, color="#555577")

        # ════════════════════════════════════════════════
        # ② 스택 영역
        # ════════════════════════════════════════════════
        tot = scores["beginning"] + scores["prompt"] + scores["summary"] + 1e-8
        ax_stack.stackplot(
            steps,
            scores["beginning"] / tot,
            scores["prompt"]    / tot,
            scores["summary"]   / tot,
            labels=[LABELS["beginning"], LABELS["prompt"], LABELS["summary"]],
            colors=[COLORS["beginning"], COLORS["prompt"], COLORS["summary"]],
            alpha=0.82,
        )
        ax_stack.set_title("② Stacked Area: Relative Proportion of Each Token Group",
                            color="white", fontsize=12, pad=10)
        ax_stack.set_xlabel("Diffusion Step", color="#AAAACC", fontsize=10)
        ax_stack.set_ylabel("Proportion (합=1)", color="#AAAACC", fontsize=10)
        ax_stack.set_xlim(steps[0], steps[-1]); ax_stack.set_ylim(0, 1)
        ax_stack.set_xticks(steps[::xtick_step])
        ax_stack.set_xticklabels(step_labels[::xtick_step],
                                  rotation=30, ha="right", fontsize=8, color="white")
        ax_stack.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax_stack.legend(fontsize=9, loc="upper right",
                        facecolor=dark, labelcolor="white", edgecolor="#444466")
        ax_stack.grid(True, alpha=0.15, color="#555577", axis="y")

        # ════════════════════════════════════════════════
        # ③ 히트맵
        # ════════════════════════════════════════════════
        ts = target_res * target_res
        sorted_steps = sorted(self.store.step_attention.keys())
        seq_len = self.tokenizer.model_max_length  # 77

        heat = np.zeros((seq_len, len(sorted_steps)))
        for ci, step in enumerate(sorted_steps):
            maps = [a for a in self.store.step_attention[step] if a.shape[1] == ts]
            if not maps: continue
            pt = torch.stack(maps).float().mean(0).sum(1).mean(0).numpy()  # (77,)
            pt = pt / (pt.sum() + 1e-8)
            heat[:, ci] = pt[:seq_len]

        vmax = np.percentile(heat[heat > 0], 98) if heat.max() > 0 else 1
        im = ax_heat.imshow(heat, aspect="auto", cmap="hot",
                            interpolation="nearest", vmin=0, vmax=vmax)
        cb = plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
        cb.set_label("Normalized Attention", color="white")
        cb.ax.yaxis.label.set_color("white"); cb.ax.tick_params(colors="white")

        for g, color in COLORS.items():
            idx = groups[g]
            if not idx: continue
            lo, hi = min(idx) - 0.5, max(idx) + 0.5
            ax_heat.axhline(lo, color=color, lw=1.2, ls="--", alpha=0.8)
            ax_heat.axhline(hi, color=color, lw=1.2, ls="--", alpha=0.8)
            ax_heat.text(len(sorted_steps) + 0.3, (lo + hi) / 2,
                         g[:3].upper(), color=color, fontsize=7,
                         va="center", fontweight="bold")

        ytick_pos = list(range(0, seq_len, max(1, seq_len // 15)))
        ax_heat.set_yticks(ytick_pos)
        ax_heat.set_yticklabels([labels[i] for i in ytick_pos],
                                 fontsize=7, color="white")
        ax_heat.set_xticks(range(0, len(sorted_steps), xtick_step))
        ax_heat.set_xticklabels([f"T-{i}" for i in range(0, len(sorted_steps), xtick_step)],
                                  rotation=30, ha="right", fontsize=8, color="white")
        ax_heat.set_title("③ Per-Token Attention Heatmap over Steps\n"
                          "(밝을수록 attention 집중, 그룹 경계 = 점선)",
                          color="white", fontsize=12, pad=10)
        ax_heat.set_xlabel("Diffusion Step", color="#AAAACC", fontsize=10)
        ax_heat.set_ylabel("Token Position", color="#AAAACC", fontsize=10)

        # ════════════════════════════════════════════════
        # ④-A 원본 이미지
        # ════════════════════════════════════════════════
        ax_img.imshow(self.image)
        ax_img.set_title("Generated Image", color="white", fontsize=11)
        ax_img.axis("off")

        # ════════════════════════════════════════════════
        # ④-B 토큰 분류 요약표
        # ════════════════════════════════════════════════
        ax_table.axis("off")
        ax_table.set_title("Token Classification Summary", color="white", fontsize=11)

        table_data, col_hdrs = [], ["Group", "#Tokens", "Mean Attn", "Final Attn", "Tokens"]
        for g in ["beginning", "prompt", "summary"]:
            idx      = groups[g]
            preview  = ", ".join(labels[i] for i in idx[:5])
            if len(idx) > 5: preview += f" (+{len(idx)-5})"
            table_data.append([
                g.capitalize(), str(len(idx)),
                f"{np.nanmean(scores[g]):.4f}",
                f"{scores[g][-1] if len(scores[g]) > 0 else 0:.4f}",
                preview,
            ])

        tbl = ax_table.table(cellText=table_data, colLabels=col_hdrs,
                              cellLoc="center", loc="center",
                              bbox=[0, 0.2, 1, 0.7])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8)
        g_order = ["beginning", "prompt", "summary"]
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor(dark); cell.set_edgecolor("#444466")
            cell.set_text_props(color="white")
            if row == 0:
                cell.set_facecolor("#2C2F45")
                cell.set_text_props(color="white", fontweight="bold")
            elif 1 <= row <= 3:
                r, gg, b, _ = mcolors.to_rgba(COLORS[g_order[row-1]])
                cell.set_facecolor((r*0.3, gg*0.3, b*0.3, 1.0))

        ax_table.text(0.02, 0.12,
            "📌 암기 시: Summary token이 높고 느리게 감소\n"
            "📌 정상 생성: Prompt token이 지배적\n"
            "📌 Beginning token: 후반부 증가",
            transform=ax_table.transAxes, fontsize=8, color="#AAAACC",
            va="bottom",
            bbox=dict(facecolor="#2C2F45", alpha=0.6, edgecolor="#444466",
                      boxstyle="round,pad=0.4"),
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"💾 저장: {save_path}")
        else:
            plt.tight_layout()
            plt.show()

        return scores


# ─────────────────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     PROMPT = "a photo of an astronaut riding a horse on the moon"

#     viz   = SDAttentionVisualizer(device="cuda")
#     image = viz.generate(PROMPT, num_inference_steps=50, seed=42)

#     viz.visualize_per_token(save_path="attn_per_token.png")
#     viz.visualize_entropy_over_steps(save_path="entropy_steps.png")

#     scores = viz.visualize_token_group_attention_over_steps(
#         target_res=16, save_path="token_group_steps.png"
#     )
#     print(f"\n📊 평균 Attention | Beginning: {np.nanmean(scores['beginning']):.4f} "
#           f"| Prompt: {np.nanmean(scores['prompt']):.4f} "
#           f"| Summary: {np.nanmean(scores['summary']):.4f}")
# ============================================================
# sd_attention_visualizer.py  ─  완전 독립 실행 파일
# ============================================================
# ─────────────────────────────────────────────────────────────
# 핵심 계산: special/prompt token attention 분포 간 cross-entropy
# ─────────────────────────────────────────────────────────────

def _interp_to_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    1-D 배열을 target_len으로 선형 보간 후 재정규화.
    (두 분포의 크기가 다를 때 같은 공간으로 맞춰줌)
    """
    if len(arr) == target_len:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    interp = np.interp(x_new, x_old, arr)
    interp = np.clip(interp, 0, None)
    total  = interp.sum()
    return interp / (total + 1e-12)


def compute_group_cross_entropy_per_step(
    store      : "AttentionStore",
    groups     : Dict[str, List[int]],
    target_res : int = 16,
) -> Dict[str, np.ndarray]:
    """
    각 타임스텝 t에서 special / prompt token attention 분포 간 cross-entropy 계산.

    처리 흐름:
        A_t ∈ R^77  (전체 attention, normalized)
          ↓ 그룹별 추출 + 정규화
        P_S = A_t[I_S] / Σ A_t[I_S]   (special 분포, 크기 |I_S|)
        P_P = A_t[I_P] / Σ A_t[I_P]   (prompt  분포, 크기 |I_P|)
          ↓ 짧은 쪽을 긴 쪽으로 보간 (선형 보간 + 재정규화)
        CE(P_S → P_P) = -Σ P_S_i · log(P_P_i)
        CE(P_P → P_S) = -Σ P_P_i · log(P_S_i)

    Returns:
        {
          "ce_s2p": np.ndarray (n_steps,),  CE(special → prompt)
          "ce_p2s": np.ndarray (n_steps,),  CE(prompt  → special)
          "h_s"   : np.ndarray (n_steps,),  H(special) self-entropy
          "h_p"   : np.ndarray (n_steps,),  H(prompt)  self-entropy
        }
    """
    EPS = 1e-12
    ts = target_res * target_res
    steps = sorted(store.step_attention.keys())

    special_idx = groups["beginning"] + groups["summary"]
    prompt_idx  = groups["prompt"]

    ce_s2p_list, ce_p2s_list = [], []
    h_s_list,    h_p_list    = [], []

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == ts]
        if not maps:
            for lst in [ce_s2p_list, ce_p2s_list, h_s_list, h_p_list]:
                lst.append(np.nan)
            continue

        # (layers, heads, spatial, 77) → 평균 → (77,) normalized
        per_tok = torch.stack(maps).float().mean(0).sum(1).mean(0).numpy()
        per_tok = per_tok / (per_tok.sum() + EPS)

        # 각 그룹 추출 + 정규화 → 독립 분포
        v_s = per_tok[[i for i in special_idx if i < len(per_tok)]]
        v_p = per_tok[[i for i in prompt_idx  if i < len(per_tok)]]

        v_s = np.clip(v_s, 0, None); v_s = v_s / (v_s.sum() + EPS)
        v_p = np.clip(v_p, 0, None); v_p = v_p / (v_p.sum() + EPS)

        # 크기 정렬 (긴 쪽 기준으로 보간)
        target_len = max(len(v_s), len(v_p))
        p_s = _interp_to_len(v_s, target_len)   # special 분포 (보간)
        p_p = _interp_to_len(v_p, target_len)   # prompt  분포 (보간)

        # Cross-Entropy 계산
        ce_s2p = float(-np.sum(p_s * np.log(p_p + EPS)))  # CE(special → prompt)
        ce_p2s = float(-np.sum(p_p * np.log(p_s + EPS)))  # CE(prompt  → special)

        # Self-Entropy (참조용)
        h_s = float(-np.sum(p_s * np.log(p_s + EPS)))
        h_p = float(-np.sum(p_p * np.log(p_p + EPS)))

        ce_s2p_list.append(ce_s2p)
        ce_p2s_list.append(ce_p2s)
        h_s_list.append(h_s)
        h_p_list.append(h_p)

    return {
        "ce_s2p": np.array(ce_s2p_list),
        "ce_p2s": np.array(ce_p2s_list),
        "h_s"   : np.array(h_s_list),
        "h_p"   : np.array(h_p_list),
    }

def compute_cosine_similarity_per_step(
    store      : AttentionStore,
    groups     : Dict[str, List[int]],
    target_res : int = 16,
) -> np.ndarray:
    """
    각 타임스텝에서 special(beginning+summary) / prompt token
    attention 벡터 간 코사인 유사도 계산.

    special_idx = groups["beginning"] + groups["summary"]  ← BOS + EOS/PAD 전부
    prompt_idx  = groups["prompt"]
    """
    EPS = 1e-12
    ts  = target_res * target_res
    steps = sorted(store.step_attention.keys())

    # ✅ special = beginning + summary 모두 포함
    special_idx = groups["beginning"] + groups["summary"]
    prompt_idx  = groups["prompt"]

    cos_sim_list = []

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == ts]
        if not maps:
            cos_sim_list.append(np.nan)
            continue

        # (layers, heads, spatial, 77) → 평균 → (77,) normalized
        per_tok = torch.stack(maps).float().mean(0).sum(1).mean(0).numpy()
        per_tok = per_tok / (per_tok.sum() + EPS)

        v_s = per_tok[[i for i in special_idx if i < len(per_tok)]]
        v_p = per_tok[[i for i in prompt_idx  if i < len(per_tok)]]

        v_s = np.clip(v_s, 0, None)
        v_p = np.clip(v_p, 0, None)

        # 같은 길이로 보간
        L   = max(len(v_s), len(v_p), 1)
        v_s = _interp_to_len(v_s, L)
        v_p = _interp_to_len(v_p, L)

        dot  = np.dot(v_s, v_p)
        norm = np.linalg.norm(v_s) * np.linalg.norm(v_p) + EPS
        cos_sim_list.append(float(dot / norm))

    return np.array(cos_sim_list)

# ─────────────────────────────────────────────────────────────
# 시각화: 두 프롬프트를 하나의 그래프로 비교
# ─────────────────────────────────────────────────────────────

def visualize_group_ce_comparison(
    store1: AttentionStore, groups1: Dict[str, List[int]], prompt1: str, image1: Image.Image,
    store2: AttentionStore, groups2: Dict[str, List[int]], prompt2: str, image2: Image.Image,
    target_res: int = 16,
    save_path: Optional[str] = None,
):
    # ── 지표 계산 ──
    m1  = compute_group_cross_entropy_per_step(store1, groups1, target_res)
    m2  = compute_group_cross_entropy_per_step(store2, groups2, target_res)
    cs1 = compute_cosine_similarity_per_step(store1, groups1, target_res)
    cs2 = compute_cosine_similarity_per_step(store2, groups2, target_res)

    steps1 = np.arange(len(m1["ce_s2p"]))
    steps2 = np.arange(len(m2["ce_s2p"]))

    # ── 레이아웃: 4행 3열 ──
    fig = plt.figure(figsize=(18, 18))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

    # ────────────────────────────────────────────────
    # ① Cross-Entropy 비교
    # ────────────────────────────────────────────────
    ax_ce = fig.add_subplot(gs[0, :2])
    ax_ce.plot(steps1, m1["ce_s2p"], 'b-',  lw=2, label="P1 CE(special→prompt)")
    ax_ce.plot(steps1, m1["ce_p2s"], 'b--', lw=2, label="P1 CE(prompt→special)")
    ax_ce.plot(steps2, m2["ce_s2p"], 'r-',  lw=2, label="P2 CE(special→prompt)")
    ax_ce.plot(steps2, m2["ce_p2s"], 'r--', lw=2, label="P2 CE(prompt→special)")
    ax_ce.fill_between(steps1, m1["ce_s2p"], m1["ce_p2s"], alpha=0.08, color='blue')
    ax_ce.fill_between(steps2, m2["ce_s2p"], m2["ce_p2s"], alpha=0.08, color='red')
    ax_ce.set_title("① Cross-Entropy: Special Token P vs Prompt Token Q\n"
                    "CE(P→Q) = P가 Q를 설명하기 어려운 정도  |  CE(Q→P) = Q가 P를 설명하기 어려운 정도",
                    fontsize=10, fontweight='bold')
    ax_ce.set_xlabel("Diffusion Step")
    ax_ce.set_ylabel("Cross-Entropy (nats)")
    ax_ce.legend(fontsize=8)
    ax_ce.grid(alpha=0.3)

    # ────────────────────────────────────────────────
    # ② Self-Entropy
    # ────────────────────────────────────────────────
    ax_ent = fig.add_subplot(gs[1, :2])
    ax_ent.plot(steps1, m1["h_s"], 'b-',  lw=1.5, label="P1 H(special)")
    ax_ent.plot(steps1, m1["h_p"], 'b--', lw=1.5, label="P1 H(prompt)")
    ax_ent.plot(steps2, m2["h_s"], 'r-',  lw=1.5, label="P2 H(special)")
    ax_ent.plot(steps2, m2["h_p"], 'r--', lw=1.5, label="P2 H(prompt)")
    ax_ent.set_title("② Self-Entropy H(P), H(Q) — 각 토큰 그룹의 집중도\n"
                     "낮을수록 특정 토큰에 attention 집중 (memorization 신호)",
                     fontsize=10, fontweight='bold')
    ax_ent.set_xlabel("Diffusion Step")
    ax_ent.set_ylabel("Entropy (nats)")
    ax_ent.legend(fontsize=8)
    ax_ent.grid(alpha=0.3)

    # ────────────────────────────────────────────────
    # ③ CE 차이 (ΔCE)
    # ────────────────────────────────────────────────
    ax_diff = fig.add_subplot(gs[2, :2])
    diff1 = m1["ce_s2p"] - m1["ce_p2s"]
    diff2 = m2["ce_s2p"] - m2["ce_p2s"]
    ax_diff.plot(steps1, diff1, 'b-', lw=2, label="P1  CE(s→p) − CE(p→s)")
    ax_diff.plot(steps2, diff2, 'r-', lw=2, label="P2  CE(s→p) − CE(p→s)")
    ax_diff.axhline(0, color='k', lw=0.8, ls='--')
    ax_diff.fill_between(steps1, diff1, 0, where=diff1 > 0, alpha=0.15, color='blue')
    ax_diff.fill_between(steps1, diff1, 0, where=diff1 < 0, alpha=0.15, color='cyan')
    ax_diff.fill_between(steps2, diff2, 0, where=diff2 > 0, alpha=0.15, color='red')
    ax_diff.fill_between(steps2, diff2, 0, where=diff2 < 0, alpha=0.15, color='orange')
    ax_diff.set_title("③ CE(special→prompt) − CE(prompt→special)\n"
                      "음수 = special이 더 응집 | 양수 = prompt가 더 복잡",
                      fontsize=10, fontweight='bold')
    ax_diff.set_xlabel("Diffusion Step")
    ax_diff.set_ylabel("ΔCE (nats)")
    ax_diff.legend(fontsize=8)
    ax_diff.grid(alpha=0.3)

    # ────────────────────────────────────────────────
    # ④ Cosine Similarity (신규 추가)
    # ────────────────────────────────────────────────
    ax_cos = fig.add_subplot(gs[3, :2])
    ax_cos.plot(steps1, cs1, 'b-o', lw=2, ms=3, label="P1 CosSim(special, prompt)")
    ax_cos.plot(steps2, cs2, 'r-o', lw=2, ms=3, label="P2 CosSim(special, prompt)")
    ax_cos.fill_between(steps1, cs1, alpha=0.10, color='blue')
    ax_cos.fill_between(steps2, cs2, alpha=0.10, color='red')
    ax_cos.axhline(0, color='k', lw=0.8, ls='--')

    # 최솟값/최댓값 어노테이션
    for steps, cs, color, tag in [(steps1, cs1, 'blue', 'P1'), (steps2, cs2, 'red', 'P2')]:
        valid = ~np.isnan(cs)
        if valid.any():
            idx_max = steps[valid][np.argmax(cs[valid])]
            idx_min = steps[valid][np.argmin(cs[valid])]
            ax_cos.annotate(f"{tag} max={cs[valid].max():.3f}",
                            xy=(idx_max, cs[idx_max]),
                            xytext=(idx_max + 0.5, cs[idx_max] + 0.02),
                            fontsize=7, color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
            ax_cos.annotate(f"{tag} min={cs[valid].min():.3f}",
                            xy=(idx_min, cs[idx_min]),
                            xytext=(idx_min + 0.5, cs[idx_min] - 0.04),
                            fontsize=7, color=color,
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax_cos.set_title("④ Cosine Similarity: Special Token vs Prompt Token Attention\n"
                     "1에 가까울수록 두 그룹 분포가 유사 | 0에 가까울수록 직교(독립적)",
                     fontsize=10, fontweight='bold')
    ax_cos.set_xlabel("Diffusion Step")
    ax_cos.set_ylabel("Cosine Similarity")
    ax_cos.set_ylim(-0.1, 1.05)
    ax_cos.legend(fontsize=8)
    ax_cos.grid(alpha=0.3)

    # ────────────────────────────────────────────────
    # 오른쪽: 이미지 & 요약 테이블
    # ────────────────────────────────────────────────
    ax_img1 = fig.add_subplot(gs[0, 2])
    ax_img1.imshow(image1)
    ax_img1.axis('off')
    ax_img1.set_title(f"Prompt 1:\n{prompt1[:45]}{'…' if len(prompt1)>45 else ''}", fontsize=8)

    ax_img2 = fig.add_subplot(gs[1, 2])
    ax_img2.imshow(image2)
    ax_img2.axis('off')
    ax_img2.set_title(f"Prompt 2:\n{prompt2[:45]}{'…' if len(prompt2)>45 else ''}", fontsize=8)

    ax_tbl = fig.add_subplot(gs[2:, 2])
    ax_tbl.axis('off')
    tbl_data = [
        ["지표", "P1", "P2"],
        ["Mean CE(s→p)",     f"{np.nanmean(m1['ce_s2p']):.4f}", f"{np.nanmean(m2['ce_s2p']):.4f}"],
        ["Mean CE(p→s)",     f"{np.nanmean(m1['ce_p2s']):.4f}", f"{np.nanmean(m2['ce_p2s']):.4f}"],
        ["Mean H(special)",  f"{np.nanmean(m1['h_s']):.4f}",    f"{np.nanmean(m2['h_s']):.4f}"],
        ["Mean H(prompt)",   f"{np.nanmean(m1['h_p']):.4f}",    f"{np.nanmean(m2['h_p']):.4f}"],
        ["Mean CE diff",     f"{np.nanmean(diff1):.4f}",         f"{np.nanmean(diff2):.4f}"],
        # ✅ 신규 추가 행
        ["Mean CosSim",      f"{np.nanmean(cs1):.4f}",           f"{np.nanmean(cs2):.4f}"],
        ["Min  CosSim",      f"{np.nanmin(cs1):.4f}",            f"{np.nanmin(cs2):.4f}"],
        ["Max  CosSim",      f"{np.nanmax(cs1):.4f}",            f"{np.nanmax(cs2):.4f}"],
    ]
    tbl = ax_tbl.table(
        cellText=tbl_data[1:],
        colLabels=tbl_data[0],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    # CosSim 행 배경 강조
    for row_idx in [6, 7, 8]:   # Mean/Min/Max CosSim
        for col_idx in range(3):
            tbl[row_idx, col_idx].set_facecolor('#e8f4fd')
    ax_tbl.set_title("Summary", fontsize=10, fontweight='bold', pad=6)

    fig.suptitle(
        "Special ↔ Prompt Token Attention Cross-Entropy & Cosine Similarity per Timestep\n"
        f"🔵 P1: \"{prompt1[:50]}\"\n🔴 P2: \"{prompt2[:50]}\"",
        fontsize=12, fontweight='bold', y=0.995
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return m1, m2, cs1, cs2   # ← 반환값에 cs1, cs2 추가



# ─────────────────────────────────────────────────────────────
# SDAttentionVisualizer에 메서드 추가
# ─────────────────────────────────────────────────────────────

# 기존 SDAttentionVisualizer 클래스에 아래 메서드 추가:
def compare_group_cross_entropy(
    self,
    prompt1: str,
    prompt2: str,
    num_inference_steps: int = 20,
    seed: int = 42,
    target_res: int = 16,
    save_path: Optional[str] = None,
):
    # ✅ classify_tokens(tokenizer, prompt) 순서로 올바르게 호출
    print(f"\n[Step 1/2] Generating image for Prompt 1: '{prompt1}'")
    image1  = self.generate(prompt1, num_inference_steps, seed)
    store1  = deepcopy(self.store)
    groups1 = classify_tokens(self.tokenizer, prompt1)   # ← 수정됨

    print(f"\n[Step 2/2] Generating image for Prompt 2: '{prompt2}'")
    image2  = self.generate(prompt2, num_inference_steps, seed)
    store2  = deepcopy(self.store)
    groups2 = classify_tokens(self.tokenizer, prompt2)   # ← 수정됨

    # special = beginning + summary 확인 출력
    print(f"\n[P1 Token Groups]")
    print(f"  beginning : {groups1['beginning']}")
    print(f"  prompt    : {groups1['prompt']}")
    print(f"  summary   : {groups1['summary']}")
    print(f"  ✅ special : beginning({len(groups1['beginning'])}) "
          f"+ summary({len(groups1['summary'])}) "
          f"= {len(groups1['beginning']) + len(groups1['summary'])} tokens")

    print(f"\n[P2 Token Groups]")
    print(f"  beginning : {groups2['beginning']}")
    print(f"  prompt    : {groups2['prompt']}")
    print(f"  summary   : {groups2['summary']}")
    print(f"  ✅ special : beginning({len(groups2['beginning'])}) "
          f"+ summary({len(groups2['summary'])}) "
          f"= {len(groups2['beginning']) + len(groups2['summary'])} tokens")

    m1, m2, cs1, cs2 = visualize_group_ce_comparison(
        store1, groups1, prompt1, image1,
        store2, groups2, prompt2, image2,
        target_res=target_res,
        save_path=save_path,
    )

    diff1 = m1["ce_s2p"] - m1["ce_p2s"]
    diff2 = m2["ce_s2p"] - m2["ce_p2s"]
    print("\n" + "=" * 60)
    print(f"{'지표':<24} {'Prompt 1':>16} {'Prompt 2':>16}")
    print("=" * 60)
    for key, label in [("ce_s2p", "CE(special→prompt)"),
                        ("ce_p2s", "CE(prompt→special)"),
                        ("h_s",    "H(special)"),
                        ("h_p",    "H(prompt)")]:
        print(f"{label:<24} {np.nanmean(m1[key]):>16.4f} {np.nanmean(m2[key]):>16.4f}")
    print(f"{'CE diff':<24} {np.nanmean(diff1):>16.4f} {np.nanmean(diff2):>16.4f}")
    print("-" * 60)
    print(f"{'CosSim Mean':<24} {np.nanmean(cs1):>16.4f} {np.nanmean(cs2):>16.4f}")
    print(f"{'CosSim Min':<24} {np.nanmin(cs1):>16.4f}  {np.nanmin(cs2):>16.4f}")
    print(f"{'CosSim Max':<24} {np.nanmax(cs1):>16.4f}  {np.nanmax(cs2):>16.4f}")
    print("=" * 60)
    return m1, m2, cs1, cs2
# ─────────────────────────────────────────────────────────────
# 프롬프트 리스트 → 스텝별 지표 평균 계산
# ─────────────────────────────────────────────────────────────

def _pad_to(arr: np.ndarray, length: int) -> np.ndarray:
    """배열을 target length까지 nan으로 패딩"""
    if len(arr) >= length:
        return arr[:length]
    return np.concatenate([arr, np.full(length - len(arr), np.nan)])


def compute_metrics_for_prompt_list(
    viz,                          # SDAttentionVisualizer 인스턴스
    prompt_list     : List[str],
    num_inference_steps: int = 20,
    seed            : int  = 42,
    target_res      : int  = 16,
    group_label     : str  = "group",
) -> Dict:
    """
    프롬프트 리스트의 각 프롬프트에 대해 CE / CosSim 계산 후
    스텝별 mean ± std 반환.

    Returns:
        {
          "ce_s2p_mean", "ce_s2p_std",
          "ce_p2s_mean", "ce_p2s_std",
          "h_s_mean",    "h_s_std",
          "h_p_mean",    "h_p_std",
          "cos_sim_mean","cos_sim_std",
          "diff_mean",   "diff_std",    # CE(s→p) - CE(p→s)
          "n_prompts": int,
          "per_prompt": List[Dict],     # 개별 프롬프트 원본 값 보존
        }
    """
    all_ce_s2p, all_ce_p2s = [], []
    all_h_s,    all_h_p    = [], []
    all_cos_sim            = []
    per_prompt_records     = []

    total = len(prompt_list)
    for i, prompt in enumerate(prompt_list):
        print(f"\n  [{group_label}] {i+1}/{total}: \"{prompt[:60]}\"")
        viz.generate(prompt, num_inference_steps=num_inference_steps, seed=seed)
        store  = deepcopy(viz.store)
        groups = classify_tokens(viz.tokenizer, prompt)

        m   = compute_group_cross_entropy_per_step(store, groups, target_res)
        cs  = compute_cosine_similarity_per_step(store, groups, target_res)

        all_ce_s2p.append(m["ce_s2p"])
        all_ce_p2s.append(m["ce_p2s"])
        all_h_s.append(m["h_s"])
        all_h_p.append(m["h_p"])
        all_cos_sim.append(cs)
        per_prompt_records.append({
            "prompt": prompt, "ce_s2p": m["ce_s2p"], "ce_p2s": m["ce_p2s"],
            "h_s": m["h_s"], "h_p": m["h_p"], "cos_sim": cs,
        })

    # 스텝 수 통일 (가장 긴 것 기준, 짧은 건 nan 패딩)
    max_steps = max(len(x) for x in all_ce_s2p)

    def _stack(lst):
        return np.stack([_pad_to(x, max_steps) for x in lst])  # (N, T)

    stk_s2p = _stack(all_ce_s2p)
    stk_p2s = _stack(all_ce_p2s)
    stk_hs  = _stack(all_h_s)
    stk_hp  = _stack(all_h_p)
    stk_cs  = _stack(all_cos_sim)
    stk_dif = stk_s2p - stk_p2s

    return {
        "ce_s2p_mean" : np.nanmean(stk_s2p, axis=0),
        "ce_s2p_std"  : np.nanstd (stk_s2p, axis=0),
        "ce_p2s_mean" : np.nanmean(stk_p2s, axis=0),
        "ce_p2s_std"  : np.nanstd (stk_p2s, axis=0),
        "h_s_mean"    : np.nanmean(stk_hs,  axis=0),
        "h_s_std"     : np.nanstd (stk_hs,  axis=0),
        "h_p_mean"    : np.nanmean(stk_hp,  axis=0),
        "h_p_std"     : np.nanstd (stk_hp,  axis=0),
        "cos_sim_mean": np.nanmean(stk_cs,  axis=0),
        "cos_sim_std" : np.nanstd (stk_cs,  axis=0),
        "diff_mean"   : np.nanmean(stk_dif, axis=0),
        "diff_std"    : np.nanstd (stk_dif, axis=0),
        "n_prompts"   : total,
        "per_prompt"  : per_prompt_records,
    }
# ─────────────────────────────────────────────────────────────
# 리스트 평균 지표 시각화 (Normal vs Memorized 비교)
# ─────────────────────────────────────────────────────────────

def visualize_list_comparison(
    normal_metrics : Dict,
    mem_metrics    : Dict,
    normal_label   : str = "Normal Prompts",
    mem_label      : str = "Memorized Prompts",
    save_path      : Optional[str] = None,
):
    """
    compute_metrics_for_prompt_list() 결과 두 개를 받아
    4개 패널 + 요약 테이블을 그린다.

    패널:
      ① CE(special→prompt)  mean ± std  (두 그룹 겹쳐 그리기)
      ② CE(prompt→special)  mean ± std
      ③ Self-Entropy H(special) / H(prompt) mean ± std
      ④ Cosine Similarity    mean ± std
      ⑤ ΔCE = CE(s→p) - CE(p→s)
      [우측] 요약 테이블
    """
    nm = normal_metrics
    mm = mem_metrics
    T  = max(len(nm["ce_s2p_mean"]), len(mm["ce_s2p_mean"]))

    # 스텝 축
    sn = np.arange(len(nm["ce_s2p_mean"]))
    sm = np.arange(len(mm["ce_s2p_mean"]))

    C_NOR = "#4A90D9"   # 파란 계열 = normal
    C_MEM = "#E74C3C"   # 빨간 계열 = memorized

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35,
                            height_ratios=[1, 1, 1, 1, 1])

    def _plot_band(ax, steps, mean, std, color, label, ls="-"):
        ax.plot(steps, mean, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.18, color=color)

    # ═══════════════════════════════════════════════
    # ① CE(special → prompt)
    # ═══════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :2])
    _plot_band(ax1, sn, nm["ce_s2p_mean"], nm["ce_s2p_std"], C_NOR, f"{normal_label} (n={nm['n_prompts']})")
    _plot_band(ax1, sm, mm["ce_s2p_mean"], mm["ce_s2p_std"], C_MEM, f"{mem_label} (n={mm['n_prompts']})")
    ax1.set_title("① CE(special → prompt)  [mean ± std]\n"
                  "높을수록 special 분포가 prompt 분포와 다름",
                  fontsize=10, fontweight="bold")
    ax1.set_xlabel("Diffusion Step"); ax1.set_ylabel("Cross-Entropy (nats)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # ═══════════════════════════════════════════════
    # ② CE(prompt → special)
    # ═══════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, :2])
    _plot_band(ax2, sn, nm["ce_p2s_mean"], nm["ce_p2s_std"], C_NOR, f"{normal_label}")
    _plot_band(ax2, sm, mm["ce_p2s_mean"], mm["ce_p2s_std"], C_MEM, f"{mem_label}")
    ax2.set_title("② CE(prompt → special)  [mean ± std]\n"
                  "높을수록 prompt 분포가 special 분포와 다름",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Diffusion Step"); ax2.set_ylabel("Cross-Entropy (nats)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # ═══════════════════════════════════════════════
    # ③ Self-Entropy
    # ═══════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2, :2])
    _plot_band(ax3, sn, nm["h_s_mean"], nm["h_s_std"], C_NOR,
               f"{normal_label} H(special)", ls="-")
    _plot_band(ax3, sn, nm["h_p_mean"], nm["h_p_std"], C_NOR,
               f"{normal_label} H(prompt)",  ls="--")
    _plot_band(ax3, sm, mm["h_s_mean"], mm["h_s_std"], C_MEM,
               f"{mem_label} H(special)",    ls="-")
    _plot_band(ax3, sm, mm["h_p_mean"], mm["h_p_std"], C_MEM,
               f"{mem_label} H(prompt)",     ls="--")
    ax3.set_title("③ Self-Entropy H(special) / H(prompt)  [mean ± std]\n"
                  "낮을수록 해당 그룹 내 특정 토큰에 집중 (memorization 신호)",
                  fontsize=10, fontweight="bold")
    ax3.set_xlabel("Diffusion Step"); ax3.set_ylabel("Entropy (nats)")
    ax3.legend(fontsize=8, ncol=2); ax3.grid(alpha=0.3)

    # ═══════════════════════════════════════════════
    # ④ Cosine Similarity
    # ═══════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[3, :2])
    _plot_band(ax4, sn, nm["cos_sim_mean"], nm["cos_sim_std"], C_NOR, f"{normal_label}")
    _plot_band(ax4, sm, mm["cos_sim_mean"], mm["cos_sim_std"], C_MEM, f"{mem_label}")
    ax4.axhline(0, color="k", lw=0.8, ls="--")
    ax4.set_title("④ Cosine Similarity(special, prompt)  [mean ± std]\n"
                  "낮을수록 두 그룹 attention 분포가 다름 (memorized: 낮은 값 예상)",
                  fontsize=10, fontweight="bold")
    ax4.set_xlabel("Diffusion Step"); ax4.set_ylabel("Cosine Similarity")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # ═══════════════════════════════════════════════
    # ⑤ ΔCE = CE(s→p) - CE(p→s)
    # ═══════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[4, :2])
    _plot_band(ax5, sn, nm["diff_mean"], nm["diff_std"], C_NOR, f"{normal_label} ΔCE")
    _plot_band(ax5, sm, mm["diff_mean"], mm["diff_std"], C_MEM, f"{mem_label} ΔCE")
    ax5.axhline(0, color="k", lw=0.8, ls="--")
    ax5.fill_between(sn, nm["diff_mean"], 0,
                     where=nm["diff_mean"] < 0, alpha=0.12, color=C_NOR)
    ax5.fill_between(sm, mm["diff_mean"], 0,
                     where=mm["diff_mean"] < 0, alpha=0.12, color=C_MEM)
    ax5.set_title("⑤ ΔCE = CE(special→prompt) − CE(prompt→special)  [mean ± std]\n"
                  "음수 = special이 더 응집된 분포 | memorized일수록 더 음수 예상",
                  fontsize=10, fontweight="bold")
    ax5.set_xlabel("Diffusion Step"); ax5.set_ylabel("ΔCE (nats)")
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

    # ═══════════════════════════════════════════════
    # 우측 요약 테이블 (전체 높이 span)
    # ═══════════════════════════════════════════════
    ax_tbl = fig.add_subplot(gs[:, 2])
    ax_tbl.axis("off")

    rows = [
        ["지표", normal_label[:18], mem_label[:18], "차이(M-N)"],
        ["n_prompts",
         str(nm["n_prompts"]), str(mm["n_prompts"]), "-"],
        ["Mean CE(s→p)",
         f"{np.nanmean(nm['ce_s2p_mean']):.4f}",
         f"{np.nanmean(mm['ce_s2p_mean']):.4f}",
         f"{np.nanmean(mm['ce_s2p_mean'])-np.nanmean(nm['ce_s2p_mean']):+.4f}"],
        ["Mean CE(p→s)",
         f"{np.nanmean(nm['ce_p2s_mean']):.4f}",
         f"{np.nanmean(mm['ce_p2s_mean']):.4f}",
         f"{np.nanmean(mm['ce_p2s_mean'])-np.nanmean(nm['ce_p2s_mean']):+.4f}"],
        ["Mean H(special)",
         f"{np.nanmean(nm['h_s_mean']):.4f}",
         f"{np.nanmean(mm['h_s_mean']):.4f}",
         f"{np.nanmean(mm['h_s_mean'])-np.nanmean(nm['h_s_mean']):+.4f}"],
        ["Mean H(prompt)",
         f"{np.nanmean(nm['h_p_mean']):.4f}",
         f"{np.nanmean(mm['h_p_mean']):.4f}",
         f"{np.nanmean(mm['h_p_mean'])-np.nanmean(nm['h_p_mean']):+.4f}"],
        ["Mean ΔCE",
         f"{np.nanmean(nm['diff_mean']):.4f}",
         f"{np.nanmean(mm['diff_mean']):.4f}",
         f"{np.nanmean(mm['diff_mean'])-np.nanmean(nm['diff_mean']):+.4f}"],
        ["Mean CosSim",
         f"{np.nanmean(nm['cos_sim_mean']):.4f}",
         f"{np.nanmean(mm['cos_sim_mean']):.4f}",
         f"{np.nanmean(mm['cos_sim_mean'])-np.nanmean(nm['cos_sim_mean']):+.4f}"],
        ["Min  CosSim",
         f"{np.nanmin(nm['cos_sim_mean']):.4f}",
         f"{np.nanmin(mm['cos_sim_mean']):.4f}",
         f"{np.nanmin(mm['cos_sim_mean'])-np.nanmin(nm['cos_sim_mean']):+.4f}"],
    ]

    tbl = ax_tbl.table(
        cellText=[r[1:] for r in rows[1:]],
        rowLabels=[r[0] for r in rows[1:]],
        colLabels=rows[0][1:],
        loc="center", cellLoc="center",
        bbox=[0, 0.05, 1, 0.90],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.0, 1.8)

    # 헤더 색상
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:   # 차이 열: 양수=빨강, 음수=파랑 강조 (CosSim은 반대)
            try:
                val = float(cell.get_text().get_text())
                cell.set_facecolor("#FADBD8" if val > 0 else "#D6EAF8")
            except ValueError:
                pass

    ax_tbl.set_title("요약 테이블\n(M-N = Mem minus Normal)", fontsize=10, fontweight="bold", pad=8)

    # 프롬프트 목록 (하단)
    ax_tbl.text(0.0, 0.0,
        f"Normal prompts ({nm['n_prompts']}):\n" +
        "\n".join(f"  • {p[:55]}{'…' if len(p)>55 else ''}"
                  for p in [r['prompt'] for r in nm['per_prompt']]) +
        f"\n\nMem prompts ({mm['n_prompts']}):\n" +
        "\n".join(f"  • {p[:55]}{'…' if len(p)>55 else ''}"
                  for p in [r['prompt'] for r in mm['per_prompt']]),
        transform=ax_tbl.transAxes, fontsize=6.5, va="top",
        color="#555555",
        bbox=dict(facecolor="#F8F9FA", alpha=0.7, edgecolor="#CCCCCC",
                  boxstyle="round,pad=0.3"),
    )

    fig.suptitle(
        f"Normal vs Memorized Prompts — Cross-Attention Metrics per Timestep\n"
        f"(Normal n={nm['n_prompts']}  |  Memorized n={mm['n_prompts']}  |  "
        f"shaded = ±1 std)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return nm, mm
# ─────────────────────────────────────────────────────────────
# 리스트 기반 비교 메서드
# ─────────────────────────────────────────────────────────────

def compare_prompt_lists(
    self,
    normal_prompts     : List[str],
    mem_prompts        : List[str],
    num_inference_steps: int  = 20,
    seed               : int  = 42,
    target_res         : int  = 16,
    save_path          : Optional[str] = None,
):
    """
    normal_prompts  : 정상 생성 프롬프트 리스트
    mem_prompts     : 암기(memorized) 프롬프트 리스트
    각 리스트의 모든 프롬프트에 대해 CE / CosSim 계산 후
    스텝별 mean ± std를 하나의 그래프에 비교.
    """
    print(f"\n{'='*60}")
    print(f"  Normal  prompts: {len(normal_prompts)}개")
    print(f"  Memorized prompts: {len(mem_prompts)}개")
    print(f"{'='*60}")

    print("\n[Phase 1] Normal prompts 처리 중...")
    nm = compute_metrics_for_prompt_list(
        self, normal_prompts,
        num_inference_steps=num_inference_steps,
        seed=seed, target_res=target_res,
        group_label="Normal",
    )

    print("\n[Phase 2] Memorized prompts 처리 중...")
    mm = compute_metrics_for_prompt_list(
        self, mem_prompts,
        num_inference_steps=num_inference_steps,
        seed=seed, target_res=target_res,
        group_label="Memorized",
    )

    # 시각화
    visualize_list_comparison(
        nm, mm,
        normal_label="Normal Prompts",
        mem_label="Memorized Prompts",
        save_path=save_path,
    )

    # 콘솔 요약
    print("\n" + "=" * 65)
    print(f"{'지표':<22} {'Normal':>18} {'Memorized':>18} {'Δ(M-N)':>10}")
    print("=" * 65)
    metrics_to_print = [
        ("ce_s2p_mean", "CE(s→p)"),
        ("ce_p2s_mean", "CE(p→s)"),
        ("h_s_mean",    "H(special)"),
        ("h_p_mean",    "H(prompt)"),
        ("diff_mean",   "ΔCE"),
        ("cos_sim_mean","CosSim"),
    ]
    for key, label in metrics_to_print:
        n_val = np.nanmean(nm[key])
        m_val = np.nanmean(mm[key])
        print(f"{label:<22} {n_val:>18.4f} {m_val:>18.4f} {m_val-n_val:>+10.4f}")
    print("=" * 65)
    return nm, mm

# 클래스에 바인딩
SDAttentionVisualizer.compare_prompt_lists = compare_prompt_lists

# 클래스에 바인딩
SDAttentionVisualizer.compare_group_cross_entropy = compare_group_cross_entropy
# ─────────────────────────────────────────────────────────────
# Summary/Prompt Attention Ratio 계산 (BOS 제외)
# ─────────────────────────────────────────────────────────────

def compute_summary_prompt_ratio_per_step(
    store      : AttentionStore,
    groups     : Dict[str, List[int]],
    target_res : int = 16,
) -> np.ndarray:
    """
    각 타임스텝 t 에서:
        ratio(t) = Σ A_t[I_summary] / Σ A_t[I_prompt]

    ✅ I_summary = EOS + PAD  (beginning/BOS 제외)
    ✅ I_prompt  = 실제 단어 토큰

    값 해석:
      > 1.0  → summary 가 prompt 보다 더 많은 attention 보유 (memorization 신호)
      ≈ 1.0  → 비슷한 비율
      < 1.0  → prompt 가 summary 보다 우세 (정상 생성)
    """
    EPS = 1e-12
    ts  = target_res * target_res
    steps = sorted(store.step_attention.keys())

    summary_idx = groups["summary"]   # ← EOS/PAD만 (BOS 제외)
    prompt_idx  = groups["prompt"]

    ratio_list = []

    for step in steps:
        maps = [a for a in store.step_attention[step] if a.shape[1] == ts]
        if not maps:
            ratio_list.append(np.nan)
            continue

        # (layers, heads, spatial, 77) → 평균 → (77,) normalized (전체 합=1)
        per_tok = torch.stack(maps).float().mean(0).sum(1).mean(0).numpy()
        per_tok = per_tok / (per_tok.sum() + EPS)

        sum_s = float(per_tok[[i for i in summary_idx if i < len(per_tok)]].sum())
        sum_p = float(per_tok[[i for i in prompt_idx  if i < len(per_tok)]].sum())

        ratio_list.append(sum_s / (sum_p + EPS))

    return np.array(ratio_list)
def compute_metrics_for_prompt_list(
    viz,
    prompt_list        : List[str],
    num_inference_steps: int = 20,
    seed               : int = 42,
    target_res         : int = 16,
    group_label        : str = "group",
) -> Dict:
    all_ce_s2p, all_ce_p2s = [], []
    all_h_s,    all_h_p    = [], []
    all_cos_sim            = []
    all_ratio              = []          # ✅ 추가
    per_prompt_records     = []

    total = len(prompt_list)
    for i, prompt in enumerate(prompt_list):
        print(f"\n  [{group_label}] {i+1}/{total}: \"{prompt[:60]}\"")
        viz.generate(prompt, num_inference_steps=num_inference_steps, seed=seed)
        store  = deepcopy(viz.store)
        groups = classify_tokens(viz.tokenizer, prompt)

        m     = compute_group_cross_entropy_per_step(store, groups, target_res)
        cs    = compute_cosine_similarity_per_step(store, groups, target_res)
        ratio = compute_summary_prompt_ratio_per_step(store, groups, target_res)  # ✅ 추가

        all_ce_s2p.append(m["ce_s2p"])
        all_ce_p2s.append(m["ce_p2s"])
        all_h_s.append(m["h_s"])
        all_h_p.append(m["h_p"])
        all_cos_sim.append(cs)
        all_ratio.append(ratio)          # ✅ 추가
        per_prompt_records.append({
            "prompt": prompt, "ce_s2p": m["ce_s2p"], "ce_p2s": m["ce_p2s"],
            "h_s": m["h_s"], "h_p": m["h_p"], "cos_sim": cs,
            "ratio": ratio,              # ✅ 추가
        })

    max_steps = max(len(x) for x in all_ce_s2p)

    def _stack(lst):
        return np.stack([_pad_to(x, max_steps) for x in lst])

    stk_s2p   = _stack(all_ce_s2p)
    stk_p2s   = _stack(all_ce_p2s)
    stk_hs    = _stack(all_h_s)
    stk_hp    = _stack(all_h_p)
    stk_cs    = _stack(all_cos_sim)
    stk_dif   = stk_s2p - stk_p2s
    stk_ratio = _stack(all_ratio)        # ✅ 추가

    return {
        "ce_s2p_mean"  : np.nanmean(stk_s2p,   axis=0),
        "ce_s2p_std"   : np.nanstd (stk_s2p,   axis=0),
        "ce_p2s_mean"  : np.nanmean(stk_p2s,   axis=0),
        "ce_p2s_std"   : np.nanstd (stk_p2s,   axis=0),
        "h_s_mean"     : np.nanmean(stk_hs,    axis=0),
        "h_s_std"      : np.nanstd (stk_hs,    axis=0),
        "h_p_mean"     : np.nanmean(stk_hp,    axis=0),
        "h_p_std"      : np.nanstd (stk_hp,    axis=0),
        "cos_sim_mean" : np.nanmean(stk_cs,    axis=0),
        "cos_sim_std"  : np.nanstd (stk_cs,    axis=0),
        "diff_mean"    : np.nanmean(stk_dif,   axis=0),
        "diff_std"     : np.nanstd (stk_dif,   axis=0),
        "ratio_mean"   : np.nanmean(stk_ratio, axis=0),  # ✅ 추가
        "ratio_std"    : np.nanstd (stk_ratio, axis=0),  # ✅ 추가
        "ratio_per_prompt": stk_ratio,                   # ✅ 개별 곡선 보존
        "n_prompts"    : total,
        "per_prompt"   : per_prompt_records,
    }
# ─────────────────────────────────────────────────────────────
# Summary/Prompt Ratio 전용 시각화
# ─────────────────────────────────────────────────────────────

def visualize_summary_prompt_ratio(
    normal_metrics : Dict,
    mem_metrics    : Dict,
    normal_label   : str  = "Normal Prompts",
    mem_label      : str  = "Memorized Prompts",
    save_path      : Optional[str] = None,
    show_individual: bool = True,   # 개별 프롬프트 선도 함께 표시
):
    """
    ratio(t) = Σ A_t[EOS/PAD] / Σ A_t[prompt words]
    타임스텝별 Normal vs Memorized 평균 비율 비교 그래프.

    서브플롯:
      ① ratio mean ± std  (메인)
      ② ratio 절대값 차이 (Mem − Normal)
      ③ (옵션) 개별 프롬프트 얇은 선
      우측: 요약 수치 테이블
    """
    nm = normal_metrics
    mm = mem_metrics

    sn = np.arange(len(nm["ratio_mean"]))
    sm = np.arange(len(mm["ratio_mean"]))

    C_NOR  = "#4A90D9"
    C_MEM  = "#E74C3C"
    C_DIFF = "#F39C12"

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(
        3 if show_individual else 2, 3,
        figure=fig, hspace=0.50, wspace=0.30,
        height_ratios=[2, 1, 1.5] if show_individual else [2, 1],
    )

    # ══════════════════════════════════════════════
    # ① 메인: ratio mean ± std
    # ══════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :2])

    # ─ mean 곡선 + ±std 밴드 ─
    ax1.plot(sn, nm["ratio_mean"], color=C_NOR, lw=2.5,
             label=f"{normal_label} (n={nm['n_prompts']})")
    ax1.fill_between(sn,
                     nm["ratio_mean"] - nm["ratio_std"],
                     nm["ratio_mean"] + nm["ratio_std"],
                     alpha=0.20, color=C_NOR)

    ax1.plot(sm, mm["ratio_mean"], color=C_MEM, lw=2.5,
             label=f"{mem_label} (n={mm['n_prompts']})")
    ax1.fill_between(sm,
                     mm["ratio_mean"] - mm["ratio_std"],
                     mm["ratio_mean"] + mm["ratio_std"],
                     alpha=0.20, color=C_MEM)

    ax1.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.7,
                label="ratio = 1.0 (균형)")

    # ─ 각 그룹의 최대 시점 표시 ─
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
                xytext=(peak_idx + max(1, len(steps_arr)//10), peak_val * 1.05),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

    ax1.set_title(
        "① Summary(EOS/PAD) ÷ Prompt Token Attention Ratio  [mean ± std]\n"
        "ratio > 1 → EOS/PAD가 prompt보다 더 많은 attention 보유  |  "
        "memorized 프롬프트일수록 높을 것으로 예상",
        fontsize=11, fontweight="bold",
    )
    ax1.set_xlabel("Diffusion Step")
    ax1.set_ylabel("ratio  =  Σ A[EOS/PAD] / Σ A[prompt]")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # ══════════════════════════════════════════════
    # ② 차이: Mem − Normal
    # ══════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, :2])

    # 두 곡선의 스텝 수가 다를 수 있으므로 공통 길이 사용
    common_len = min(len(nm["ratio_mean"]), len(mm["ratio_mean"]))
    delta = mm["ratio_mean"][:common_len] - nm["ratio_mean"][:common_len]
    steps_c = np.arange(common_len)

    ax2.plot(steps_c, delta, color=C_DIFF, lw=2, label="Mem − Normal")
    ax2.fill_between(steps_c, delta, 0,
                     where=delta > 0, alpha=0.25, color=C_MEM,
                     label="Mem > Normal")
    ax2.fill_between(steps_c, delta, 0,
                     where=delta <= 0, alpha=0.25, color=C_NOR,
                     label="Normal ≥ Mem")
    ax2.axhline(0, color="k", lw=0.8, ls="--")
    ax2.set_title(
        "② Δratio = Mem − Normal  |  양수 구간 = memorized가 더 높은 비율",
        fontsize=10, fontweight="bold",
    )
    ax2.set_xlabel("Diffusion Step")
    ax2.set_ylabel("Δratio")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ══════════════════════════════════════════════
    # ③ 개별 프롬프트 선 (선택)
    # ══════════════════════════════════════════════
    if show_individual:
        ax3 = fig.add_subplot(gs[2, :2])

        # normal 개별 선
        for row in nm["ratio_per_prompt"]:
            steps_i = np.arange(len(row))
            ax3.plot(steps_i, row, color=C_NOR, lw=0.8, alpha=0.35)
        ax3.plot(sn, nm["ratio_mean"], color=C_NOR, lw=2.5,
                 label=f"{normal_label} mean")

        # mem 개별 선
        for row in mm["ratio_per_prompt"]:
            steps_i = np.arange(len(row))
            ax3.plot(steps_i, row, color=C_MEM, lw=0.8, alpha=0.35)
        ax3.plot(sm, mm["ratio_mean"], color=C_MEM, lw=2.5,
                 label=f"{mem_label} mean")

        ax3.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.6)
        ax3.set_title(
            "③ 개별 프롬프트 ratio 곡선  (연한 선 = 개별, 진한 선 = mean)",
            fontsize=10, fontweight="bold",
        )
        ax3.set_xlabel("Diffusion Step")
        ax3.set_ylabel("ratio")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

    # ══════════════════════════════════════════════
    # 우측: 요약 수치 테이블
    # ══════════════════════════════════════════════
    ax_tbl = fig.add_subplot(gs[:, 2])
    ax_tbl.axis("off")

    n_ratio = nm["ratio_mean"]
    m_ratio = mm["ratio_mean"]

    rows = [
        ["지표",               normal_label[:16],                  mem_label[:16],                    "Δ(M−N)"],
        ["n_prompts",          str(nm["n_prompts"]),               str(mm["n_prompts"]),               "—"],
        ["Mean ratio",         f"{np.nanmean(n_ratio):.4f}",       f"{np.nanmean(m_ratio):.4f}",
         f"{np.nanmean(m_ratio)-np.nanmean(n_ratio):+.4f}"],
        ["Max  ratio",         f"{np.nanmax(n_ratio):.4f}",        f"{np.nanmax(m_ratio):.4f}",
         f"{np.nanmax(m_ratio)-np.nanmax(n_ratio):+.4f}"],
        ["Min  ratio",         f"{np.nanmin(n_ratio):.4f}",        f"{np.nanmin(m_ratio):.4f}",
         f"{np.nanmin(m_ratio)-np.nanmin(n_ratio):+.4f}"],
        ["Std  ratio",         f"{np.nanstd(n_ratio):.4f}",        f"{np.nanstd(m_ratio):.4f}",       "—"],
        ["Step@peak ratio",
         str(int(np.nanargmax(n_ratio))),
         str(int(np.nanargmax(m_ratio))),                                                              "—"],
        ["Mean Δratio (M−N)",  "—",                                "—",
         f"{np.nanmean(delta):+.4f}"],
    ]

    tbl = ax_tbl.table(
        cellText=[r[1:] for r in rows[1:]],
        rowLabels=[r[0] for r in rows[1:]],
        colLabels=rows[0][1:],
        loc="upper center", cellLoc="center",
        bbox=[0, 0.45, 1, 0.50],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.scale(1.0, 1.9)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2:   # Δ 열 색상
            try:
                val = float(cell.get_text().get_text().replace("−", "-"))
                cell.set_facecolor("#FADBD8" if val > 0 else "#D6EAF8")
            except ValueError:
                pass

    ax_tbl.set_title(
        "수치 요약\n(ratio = EOS/PAD ÷ prompt)\nBOS 제외",
        fontsize=10, fontweight="bold", pad=8,
    )

    # 프롬프트 목록
    ax_tbl.text(
        0.0, 0.40,
        f"Normal ({nm['n_prompts']}):\n" +
        "\n".join(f"  {p['prompt'][:50]}{'…' if len(p['prompt'])>50 else ''}"
                  for p in nm["per_prompt"]) +
        f"\n\nMem ({mm['n_prompts']}):\n" +
        "\n".join(f"  {p['prompt'][:50]}{'…' if len(p['prompt'])>50 else ''}"
                  for p in mm["per_prompt"]),
        transform=ax_tbl.transAxes, fontsize=6, va="top", color="#444444",
        bbox=dict(facecolor="#F4F6F7", alpha=0.8, edgecolor="#CCCCCC",
                  boxstyle="round,pad=0.3"),
    )

    fig.suptitle(
        "Summary(EOS/PAD) vs Prompt Token Attention Ratio\n"
        f"Normal n={nm['n_prompts']}  |  Memorized n={mm['n_prompts']}  |  "
        "shaded = ±1 std  |  BOS 제외",
        fontsize=13, fontweight="bold", y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import matplotlib; matplotlib.use("Agg")

#     PROMPT_1 = "a photo of an astronaut riding a horse on the moon"
#     PROMPT_2 = "Photo pour Japanese pagoda and old house in Kyoto at twilight - image libre de droit"

#     viz = SDAttentionVisualizer(device="cuda")

#     result = viz.compare_group_cross_entropy(
#         prompt1=PROMPT_1,
#         prompt2=PROMPT_2,
#         num_inference_steps=50,
#         seed=42,
#         target_res=16,
#         save_path="group_ce_comparison.png",
#     )

# ╔══════════════════════════════════════════════════════════════╗
# ║  18. Value-Weighted Attention  (a_i · v_i)  Per-Head 분석  ║
# ╚══════════════════════════════════════════════════════════════╝
#
#  수식:
#    z_special(h) = mean_s [ Σ_{i∈I_special} a_h[s,i] · V_h[i,:] ]   shape: (head_dim,)
#    z_prompt(h)  = mean_s [ Σ_{j∈I_prompt}  a_h[s,j] · V_h[j,:] ]   shape: (head_dim,)
#    cos_sim(h)   = cosine( z_special(h), z_prompt(h) )
#
#  per step, per head, averaged over all cross-attention layers
# ──────────────────────────────────────────────────────────────

from typing import Tuple   # already imported if using Python 3.8; add if missing

# ─────────────────────────────────────────────────────────────
# 18-A. 확장된 AttentionStore (step_hv 필드 추가)
#   ※ 기존 AttentionStore 클래스를 아래 클래스로 교체하거나
#      __init__/reset 에 step_hv 두 줄만 추가해도 됩니다.
# ─────────────────────────────────────────────────────────────

# 기존 AttentionStore를 다음으로 교체:
class AttentionStore:
    def __init__(self):
        self.step_attention: Dict[int, List[torch.Tensor]] = defaultdict(list)
        # ▼ NEW: per-step, per-layer weighted value 저장
        #   step -> list of (av_special, av_prompt)
        #   av_special / av_prompt : np.ndarray  (n_heads, head_dim)
        self.step_hv: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        self.step_counter: int = 0

    def reset(self):
        self.step_attention = defaultdict(list)
        self.step_hv        = defaultdict(list)   # ▼ NEW
        self.step_counter   = 0


# ─────────────────────────────────────────────────────────────
# 18-B. 글로벌 토큰 인덱스 (ValueAwareProcessor 가 참조)
# ─────────────────────────────────────────────────────────────

_HV_SPECIAL_IDX: List[int] = []
_HV_PROMPT_IDX:  List[int] = []


# ─────────────────────────────────────────────────────────────
# 18-C. _ValueAwareAttnProcessor
#        기존 AttnProcessor 를 완전히 대체 (표준 연산 + HV 캡쳐)
# ─────────────────────────────────────────────────────────────

class _ValueAwareAttnProcessor:
    """
    diffusers 표준 AttnProcessor 와 동일한 forward 연산을 수행하되,
    텍스트 cross-attention(seq=77)에서 per-head weighted value sum 을 캡쳐한다.

    캡쳐 조건:
      • _ACTIVE_STORE is not None
      • encoder_hidden_states.shape[1] == 77  (텍스트 cross-attention)
      • _HV_SPECIAL_IDX, _HV_PROMPT_IDX 가 비어 있지 않음
    """

    def __call__(
        self,
        attn,
        hidden_states:          torch.Tensor,
        encoder_hidden_states:  Optional[torch.Tensor] = None,
        attention_mask:         Optional[torch.Tensor] = None,
        temb:                   Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        global _ACTIVE_STORE, _HV_SPECIAL_IDX, _HV_PROMPT_IDX

        # ── 전처리 ────────────────────────────────────────────
        residual   = hidden_states
        is_cross   = encoder_hidden_states is not None
        is_txt_ca  = is_cross and (encoder_hidden_states.shape[1] == 77)

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        batch_size, seq_q, _ = hidden_states.shape
        seq_kv = (encoder_hidden_states.shape[1]
                  if encoder_hidden_states is not None else seq_q)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, seq_kv, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        # ── Q / K / V 계산 ────────────────────────────────────
        query = attn.to_q(hidden_states)

        _enc = encoder_hidden_states if is_cross else hidden_states
        if is_cross and getattr(attn, "norm_cross", False):
            _enc = attn.norm_encoder_hidden_states(_enc)

        key   = attn.to_k(_enc)
        value = attn.to_v(_enc)          # (B, seq_kv, dim)

        # head_to_batch_dim → (B*H, seq, head_dim)
        query_b = attn.head_to_batch_dim(query)
        key_b   = attn.head_to_batch_dim(key)
        value_b = attn.head_to_batch_dim(value)   # (B*H, seq_kv, d)

        # ── Attention weights ─────────────────────────────────
        # get_attention_scores 는 기존 패치(_install_attention_patch)가
        # 이미 후킹하여 _ACTIVE_STORE.step_attention 에 저장함.
        attn_probs = attn.get_attention_scores(
            query_b, key_b, attention_mask)        # (B*H, seq_q, seq_kv)

        # ── ▼ HV 캡쳐 ─────────────────────────────────────────
        if (
            _ACTIVE_STORE is not None
            and is_txt_ca
            and _HV_SPECIAL_IDX
            and _HV_PROMPT_IDX
        ):
            try:
                with torch.no_grad():
                    BH, sq, sk = attn_probs.shape
                    n_heads  = attn.heads
                    B_act    = BH // n_heads
                    head_dim = value_b.shape[-1]

                    # CFG: cond batch 는 두 번째 절반
                    half = B_act // 2
                    s_b  = half * n_heads if half > 0 else 0

                    # (H, sq, sk) — cond 부분만 평균
                    ap = (attn_probs[s_b:]
                          .reshape(-1, n_heads, sq, sk)
                          .mean(0).float())          # (H, sq, sk)

                    vb = (value_b[s_b:]
                          .reshape(-1, n_heads, sk, head_dim)
                          .mean(0).float())          # (H, sk, d)

                    s_idx = [i for i in _HV_SPECIAL_IDX if i < sk]
                    p_idx = [i for i in _HV_PROMPT_IDX  if i < sk]

                    if s_idx and p_idx:
                        # z_special(h) = mean_s [ Σ_i a[h,s,i]*v[h,i,:] ]
                        #              = (H, sq, |s|) @ (H, |s|, d)  → mean(sq)
                        ap_s = ap[:, :, s_idx]       # (H, sq, |s|)
                        vb_s = vb[:, s_idx, :]       # (H, |s|, d)
                        ap_p = ap[:, :, p_idx]       # (H, sq, |p|)
                        vb_p = vb[:, p_idx, :]       # (H, |p|, d)

                        av_s = torch.bmm(
                            ap_s.reshape(n_heads, sq, len(s_idx)),
                            vb_s.reshape(n_heads, len(s_idx), head_dim)
                        ).mean(1)                    # (H, d)

                        av_p = torch.bmm(
                            ap_p.reshape(n_heads, sq, len(p_idx)),
                            vb_p.reshape(n_heads, len(p_idx), head_dim)
                        ).mean(1)                    # (H, d)

                        step = _ACTIVE_STORE.step_counter
                        _ACTIVE_STORE.step_hv[step].append((
                            av_s.cpu().numpy(),      # (H, d)
                            av_p.cpu().numpy(),      # (H, d)
                        ))
            except Exception as _e:
                pass   # 캡쳐 실패는 무시 (연산 결과에 영향 없음)

        # ── 표준 출력 계산 ─────────────────────────────────────
        hidden_states = torch.bmm(attn_probs, value_b)   # (B*H, sq, d)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = (hidden_states
                             .transpose(-1, -2)
                             .reshape(B, C, H, W))

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / getattr(
            attn, "rescale_output_factor", 1.0)

        return hidden_states


# ─────────────────────────────────────────────────────────────
# 18-D. Value-Aware Processor 설치 헬퍼
# ─────────────────────────────────────────────────────────────

def _install_value_aware_processor(unet) -> None:
    """
    _force_standard_attn_processor 대신 호출.
    _ValueAwareAttnProcessor 는 표준 AttnProcessor 를 완전히 대체하며
    기존 get_attention_scores 패치와 호환된다.
    """
    try:
        unet.set_attn_processor(_ValueAwareAttnProcessor())
        print("🔧 ValueAwareAttnProcessor 설치 완료 (HV 캡쳐 활성)")
    except Exception as e:
        print(f"⚠️ ValueAwareAttnProcessor 설치 실패, 표준 프로세서로 대체: {e}")
        try:
            from diffusers.models.attention_processor import AttnProcessor
            unet.set_attn_processor(AttnProcessor())
        except Exception:
            pass


def _set_hv_groups(groups: Dict[str, List[int]]) -> None:
    """generate() 에서 프롬프트 토큰 그룹을 전역 변수에 반영."""
    global _HV_SPECIAL_IDX, _HV_PROMPT_IDX
    _HV_SPECIAL_IDX = groups["beginning"] + groups["summary"]
    _HV_PROMPT_IDX  = groups["prompt"]


# ─────────────────────────────────────────────────────────────
# 18-E. 스텝별 헤드별 코사인 유사도 계산
# ─────────────────────────────────────────────────────────────

def compute_head_av_cosine_per_step(
    store     : AttentionStore,
    eps       : float = 1e-12,
    n_heads_target: Optional[int] = None,   # None → 가장 많은 헤드 수 자동 선택
) -> np.ndarray:
    """
    Returns:  (n_steps, n_heads)  — 스텝별 헤드별 cos_sim
    
    각 스텝에서 cross-attention layer 들의 cos_sim 을 평균냄.
    n_heads_target 을 지정하면 해당 헤드 수를 가진 레이어만 사용.
    """
    steps = sorted(store.step_hv.keys())
    if not steps:
        return np.array([])

    # n_heads 자동 감지
    if n_heads_target is None:
        from collections import Counter
        cnt = Counter(av_s.shape[0]
                      for step_list in store.step_hv.values()
                      for (av_s, _) in step_list)
        n_heads_target = cnt.most_common(1)[0][0] if cnt else 8

    results = []
    for step in steps:
        layer_sims = []
        for av_s, av_p in store.step_hv[step]:
            if av_s.shape[0] != n_heads_target:
                continue   # 다른 헤드 수의 레이어는 스킵
            H, d = av_s.shape
            cos  = np.zeros(H)
            for h in range(H):
                dot  = float(np.dot(av_s[h], av_p[h]))
                norm = float(np.linalg.norm(av_s[h])
                             * np.linalg.norm(av_p[h])) + eps
                cos[h] = dot / norm
            layer_sims.append(cos)

        if layer_sims:
            results.append(np.nanmean(layer_sims, axis=0))   # (H,)
        else:
            results.append(np.full(n_heads_target, np.nan))

    return np.array(results)   # (n_steps, n_heads)


def _pad2d(arr: np.ndarray, target_steps: int) -> np.ndarray:
    """(n_steps, n_heads) 배열을 target_steps 로 패딩 (nan 채움)."""
    s, h = arr.shape
    if s >= target_steps:
        return arr[:target_steps]
    pad = np.full((target_steps - s, h), np.nan)
    return np.concatenate([arr, pad], axis=0)


# ─────────────────────────────────────────────────────────────
# 18-F. 헤드별 HV CosSim 시각화
# ─────────────────────────────────────────────────────────────

def visualize_head_av_cosine(
    cos_normal : np.ndarray,   # (n_steps, n_heads)
    cos_mem    : np.ndarray,   # (n_steps, n_heads)
    normal_label: str = "Normal Prompts",
    mem_label   : str = "Memorized Prompts",
    save_path   : Optional[str] = None,
):
    """
    5-panel 시각화:
    ①② Normal / Memorized  step×head 히트맵
    ③   Difference 히트맵  (Mem − Normal)
    ④   헤드 평균, 스텝별 선 그래프
    ⑤   스텝 평균, 헤드별 막대 그래프
    """
    n_steps_n, n_heads = cos_normal.shape
    n_steps_m          = cos_mem.shape[0]
    C_NOR = "#4A90D9"; C_MEM = "#E74C3C"

    fig = plt.figure(figsize=(22, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.50, wspace=0.32,
                            height_ratios=[1.2, 1.4, 1.0])

    def _heatmap(ax, data, title, cmap="RdYlGn", vmin=-1, vmax=1):
        im = ax.imshow(data.T, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                     label="Cosine Similarity")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Diffusion Step", fontsize=9)
        ax.set_ylabel("Head Index",     fontsize=9)
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
        # x 틱 간략화
        step_arr = np.arange(data.shape[0])
        xt = step_arr[::max(1, len(step_arr) // 8)]
        ax.set_xticks(xt); ax.set_xticklabels(xt, fontsize=7)

    # ① Normal 히트맵
    ax1 = fig.add_subplot(gs[0, 0])
    _heatmap(ax1, cos_normal,
             f"① {normal_label}\nCosSim( Σaᵢvᵢ[special],  Σaⱼvⱼ[prompt] )  per head")

    # ② Memorized 히트맵
    ax2 = fig.add_subplot(gs[0, 1])
    _heatmap(ax2, cos_mem,
             f"② {mem_label}\nCosSim( Σaᵢvᵢ[special],  Σaⱼvⱼ[prompt] )  per head")

    # ③ 차이 히트맵 (Mem − Normal)
    ax3 = fig.add_subplot(gs[1, :])
    common = min(n_steps_n, n_steps_m)
    diff   = cos_mem[:common] - cos_normal[:common]   # (common, H)
    vabs   = np.nanmax(np.abs(diff)) + 1e-6
    im3 = ax3.imshow(diff.T, aspect="auto", cmap="RdBu_r",
                     vmin=-vabs, vmax=vabs, interpolation="nearest")
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.015, pad=0.01)
    cb3.set_label("Δ CosSim  (Mem − Normal)")
    ax3.set_title("③ Δ CosSim = Mem − Normal   "
                  "(🔴 Mem 높음 = memorized 쪽이 special·prompt value 더 유사)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Diffusion Step", fontsize=9)
    ax3.set_ylabel("Head Index",     fontsize=9)
    ax3.set_yticks(range(n_heads))
    ax3.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
    xt3 = np.arange(common)[::max(1, common // 10)]
    ax3.set_xticks(xt3); ax3.set_xticklabels(xt3, fontsize=7)

    # ④ 헤드 평균 per step — 선 그래프
    ax4 = fig.add_subplot(gs[2, 0])
    sn  = np.arange(n_steps_n); sm = np.arange(n_steps_m)
    mn  = np.nanmean(cos_normal, axis=1); stn = np.nanstd(cos_normal, axis=1)
    mm  = np.nanmean(cos_mem,    axis=1); stm = np.nanstd(cos_mem,    axis=1)

    ax4.plot(sn, mn, color=C_NOR, lw=2,
             label=f"{normal_label}  (mean±std  over heads)")
    ax4.fill_between(sn, mn - stn, mn + stn, alpha=0.18, color=C_NOR)
    ax4.plot(sm, mm, color=C_MEM, lw=2, label=f"{mem_label}")
    ax4.fill_between(sm, mm - stm, mm + stm, alpha=0.18, color=C_MEM)
    ax4.axhline(0, color="k", lw=0.8, ls="--")

    # 최대/최소 어노테이션
    for steps_a, mean_a, color, tag in [
        (sn, mn, C_NOR, "N"), (sm, mm, C_MEM, "M")
    ]:
        valid = ~np.isnan(mean_a)
        if valid.any():
            pk = steps_a[valid][np.argmax(mean_a[valid])]
            ax4.annotate(f"{tag} peak\n{mean_a[valid].max():.3f}",
                         xy=(pk, mean_a[pk]),
                         xytext=(pk + max(1, len(steps_a)//10),
                                 mean_a[pk] + 0.04),
                         fontsize=7, color=color,
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax4.set_title("④ 헤드 평균 CosSim  per step  [mean ± std]",
                  fontsize=10, fontweight="bold")
    ax4.set_xlabel("Diffusion Step"); ax4.set_ylabel("Cosine Similarity")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)
    ax4.set_ylim(-0.15, 1.05)

    # ⑤ 스텝 평균 per head — 막대 그래프
    ax5 = fig.add_subplot(gs[2, 1])
    mphn = np.nanmean(cos_normal, axis=0)   # (H,)
    mphm = np.nanmean(cos_mem,    axis=0)   # (H,)
    errn = np.nanstd(cos_normal, axis=0)
    errm = np.nanstd(cos_mem,    axis=0)
    x = np.arange(n_heads); w = 0.38
    ax5.bar(x - w/2, mphn, width=w, color=C_NOR, alpha=0.85,
            yerr=errn, capsize=3, label=f"{normal_label}")
    ax5.bar(x + w/2, mphm, width=w, color=C_MEM, alpha=0.85,
            yerr=errm, capsize=3, label=f"{mem_label}")
    ax5.axhline(0, color="k", lw=0.8)

    # 가장 차이가 큰 헤드 강조
    delta_per_head = mphm - mphn
    top_head = int(np.nanargmax(np.abs(delta_per_head)))
    ax5.axvspan(top_head - 0.5, top_head + 0.5,
                alpha=0.12, color="gold", label=f"H{top_head}: Δ={delta_per_head[top_head]:+.3f}")

    ax5.set_title("⑤ 스텝 평균 CosSim  per head  (mean ± std  over steps)",
                  fontsize=10, fontweight="bold")
    ax5.set_xlabel("Head Index"); ax5.set_ylabel("Mean Cosine Similarity")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=7)
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Per-Head  CosSim(  Σᵢ∈special aᵢ·Vᵢ ,  Σⱼ∈prompt aⱼ·Vⱼ  )\n"
        f"[Normal n=?  |  Memorized n=?  |  averaged over all cross-attn layers]",
        fontsize=13, fontweight="bold", y=0.995,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# 18-G. SDAttentionVisualizer 수정
#        (1) __init__ 패치 교체  (2) generate HV 셋업 추가
# ─────────────────────────────────────────────────────────────

# ── __init__ 에서 아래 두 줄을 교체 ──────────────────────────
#  기존:  _force_standard_attn_processor(self.pipe.unet)
#  변경:  _install_value_aware_processor(self.pipe.unet)
#
# 실제로는 SDAttentionVisualizer.__init__ 내부를 직접 수정하거나
# 아래처럼 __init__ 후에 프로세서를 재설치:

_orig_viz_init = SDAttentionVisualizer.__init__

def _new_viz_init(self, model_id="CompVis/stable-diffusion-v1-4",
                  device=None):
    _orig_viz_init(self, model_id=model_id, device=device)
    # 표준 AttnProcessor → ValueAwareAttnProcessor 로 교체
    _install_value_aware_processor(self.pipe.unet)

SDAttentionVisualizer.__init__ = _new_viz_init


# ── generate 에서 HV 셋업 추가 ──────────────────────────────

_orig_generate = SDAttentionVisualizer.generate

def _new_generate(self, prompt: str,
                  num_inference_steps: int = 20,
                  seed: int = 42) -> "Image.Image":
    global _ACTIVE_STORE, _HV_SPECIAL_IDX, _HV_PROMPT_IDX

    # 토큰 그룹 → 전역 인덱스 갱신 (ValueAwareProcessor 가 참조)
    _grp = classify_tokens(self.tokenizer, prompt)
    _set_hv_groups(_grp)

    # 기존 generate 호출 (내부에서 _ACTIVE_STORE 세팅 및 step_counter 관리)
    img = _orig_generate(self, prompt, num_inference_steps, seed)
    return img

SDAttentionVisualizer.generate = _new_generate


# ─────────────────────────────────────────────────────────────
# 18-H. compute_metrics_for_prompt_list 확장
#        hv_cos_mean / hv_cos_std 키 추가
# ─────────────────────────────────────────────────────────────

_orig_compute_metrics = compute_metrics_for_prompt_list

def compute_metrics_for_prompt_list(
    viz,
    prompt_list        : List[str],
    num_inference_steps: int = 20,
    seed               : int = 42,
    target_res         : int = 16,
    group_label        : str = "group",
) -> Dict:
    """기존 함수 확장: hv_cos_mean / hv_cos_std 추가."""
    result = _orig_compute_metrics(
        viz, prompt_list, num_inference_steps, seed, target_res, group_label)

    # HV cosine 수집
    all_hv = []
    for rec in result["per_prompt"]:
        prompt = rec["prompt"]
        # generate 는 이미 호출됨 → store 에 step_hv 가 채워져 있음
        # 하지만 per_prompt 에는 store 스냅샷이 없으므로, 다시 생성해야 함
        # → 대신 아래에서 별도로 재계산하는 방식 사용

    # ── 재생성 없이 step_hv 를 직접 수집하려면
    #    compute_metrics_for_prompt_list 를 완전히 재작성해야 함.
    #    여기서는 편의상 재생성 방식 사용 (시간이 2배 걸리는 단점).
    #    단, 이미 논문 실험용이라면 아래 Full 버전 (18-I) 사용 권장.
    # ───────────────────────────────────────────────────────

    return result   # hv_cos 는 18-I 전체 재작성 버전에서 포함됨


# ─────────────────────────────────────────────────────────────
# 18-I. compute_metrics_for_prompt_list (HV 포함 Full 버전)
#        기존 함수를 이 버전으로 교체하면 됩니다.
# ─────────────────────────────────────────────────────────────

def compute_metrics_for_prompt_list_full(
    viz,
    prompt_list        : List[str],
    num_inference_steps: int = 20,
    seed               : int = 42,
    target_res         : int = 16,
    group_label        : str = "group",
) -> Dict:
    """
    기존 compute_metrics_for_prompt_list 에
    per-head value-weighted cosine similarity 를 추가한 버전.

    추가 반환 키:
      hv_cos_mean  : (n_steps, n_heads)
      hv_cos_std   : (n_steps, n_heads)
    """
    all_ce_s2p, all_ce_p2s = [], []
    all_h_s,    all_h_p    = [], []
    all_cos_sim            = []
    all_ratio              = []
    all_hv_cos             = []   # ▼ NEW: list of (n_steps_i, n_heads)
    per_prompt_records     = []

    total = len(prompt_list)
    for i, prompt in enumerate(prompt_list):
        print(f"\n  [{group_label}] {i+1}/{total}: \"{prompt[:60]}\"")
        viz.generate(prompt, num_inference_steps=num_inference_steps, seed=seed)

        # 이번 generate 후 store 스냅샷
        store  = deepcopy(viz.store)
        groups = classify_tokens(viz.tokenizer, prompt)

        m     = compute_group_cross_entropy_per_step(store, groups, target_res)
        cs    = compute_cosine_similarity_per_step(store, groups, target_res)
        ratio = compute_summary_prompt_ratio_per_step(store, groups, target_res)
        hv    = compute_head_av_cosine_per_step(store)   # ▼ NEW: (n_steps, H)

        all_ce_s2p.append(m["ce_s2p"])
        all_ce_p2s.append(m["ce_p2s"])
        all_h_s.append(m["h_s"])
        all_h_p.append(m["h_p"])
        all_cos_sim.append(cs)
        all_ratio.append(ratio)
        all_hv_cos.append(hv)          # ▼ NEW

        per_prompt_records.append({
            "prompt" : prompt,
            "ce_s2p" : m["ce_s2p"], "ce_p2s": m["ce_p2s"],
            "h_s"    : m["h_s"],    "h_p"   : m["h_p"],
            "cos_sim": cs,          "ratio" : ratio,
            "hv_cos" : hv,          # ▼ NEW
        })

    # ── 스택 & 평균 ──────────────────────────────────────────
    max_steps = max(len(x) for x in all_ce_s2p)

    def _stack1d(lst):
        return np.stack([_pad_to(x, max_steps) for x in lst])

    stk_s2p   = _stack1d(all_ce_s2p)
    stk_p2s   = _stack1d(all_ce_p2s)
    stk_hs    = _stack1d(all_h_s)
    stk_hp    = _stack1d(all_h_p)
    stk_cs    = _stack1d(all_cos_sim)
    stk_dif   = stk_s2p - stk_p2s
    stk_ratio = _stack1d(all_ratio)

    # ▼ HV: (n_prompts, n_steps, n_heads)
    n_heads_hv = max((x.shape[1] for x in all_hv_cos if x.ndim == 2 and x.size > 0),
                     default=8)
    stk_hv = np.stack([
        _pad2d(x, max_steps) if (x.ndim == 2 and x.size > 0)
        else np.full((max_steps, n_heads_hv), np.nan)
        for x in all_hv_cos
    ])   # (n_prompts, n_steps, n_heads)

    return {
        "ce_s2p_mean"      : np.nanmean(stk_s2p,   axis=0),
        "ce_s2p_std"       : np.nanstd (stk_s2p,   axis=0),
        "ce_p2s_mean"      : np.nanmean(stk_p2s,   axis=0),
        "ce_p2s_std"       : np.nanstd (stk_p2s,   axis=0),
        "h_s_mean"         : np.nanmean(stk_hs,    axis=0),
        "h_s_std"          : np.nanstd (stk_hs,    axis=0),
        "h_p_mean"         : np.nanmean(stk_hp,    axis=0),
        "h_p_std"          : np.nanstd (stk_hp,    axis=0),
        "cos_sim_mean"     : np.nanmean(stk_cs,    axis=0),
        "cos_sim_std"      : np.nanstd (stk_cs,    axis=0),
        "diff_mean"        : np.nanmean(stk_dif,   axis=0),
        "diff_std"         : np.nanstd (stk_dif,   axis=0),
        "ratio_mean"       : np.nanmean(stk_ratio, axis=0),
        "ratio_std"        : np.nanstd (stk_ratio, axis=0),
        "ratio_per_prompt" : stk_ratio,
        # ▼ NEW
        "hv_cos_mean"      : np.nanmean(stk_hv,    axis=0),   # (n_steps, H)
        "hv_cos_std"       : np.nanstd (stk_hv,    axis=0),   # (n_steps, H)
        "n_prompts"        : total,
        "per_prompt"       : per_prompt_records,
    }


# ─────────────────────────────────────────────────────────────
# 18-J. compare_prompt_lists 확장 (HV 시각화 추가)
# ─────────────────────────────────────────────────────────────

def compare_prompt_lists_hv(
    self,
    normal_prompts     : List[str],
    mem_prompts        : List[str],
    num_inference_steps: int  = 20,
    seed               : int  = 42,
    target_res         : int  = 16,
    save_path          : Optional[str] = None,
):
    """
    기존 compare_prompt_lists 에 per-head HV CosSim 시각화 추가.
    저장 파일:
      save_path              → CE / entropy / ratio 종합 그래프 (기존)
      save_path_ratio.png    → EOS/PAD ratio 전용 그래프 (기존)
      save_path_hv.png       → per-head HV cosine 그래프 (NEW)
    """
    print(f"\n{'='*60}")
    print(f"  Normal    prompts: {len(normal_prompts)}개")
    print(f"  Memorized prompts: {len(mem_prompts)}개")
    print(f"{'='*60}")

    print("\n[Phase 1] Normal prompts 처리 중...")
    nm = compute_metrics_for_prompt_list_full(
        self, normal_prompts,
        num_inference_steps=num_inference_steps,
        seed=seed, target_res=target_res, group_label="Normal",
    )

    print("\n[Phase 2] Memorized prompts 처리 중...")
    mm = compute_metrics_for_prompt_list_full(
        self, mem_prompts,
        num_inference_steps=num_inference_steps,
        seed=seed, target_res=target_res, group_label="Memorized",
    )

    # ① 기존 CE / CosSim / ratio 그래프
    visualize_list_comparison(
        nm, mm,
        normal_label="Normal Prompts", mem_label="Memorized Prompts",
        save_path=save_path,
    )
    ratio_save = save_path.replace(".png", "_ratio.png") if save_path else None
    visualize_summary_prompt_ratio(
        nm, mm,
        normal_label="Normal Prompts", mem_label="Memorized Prompts",
        save_path=ratio_save, show_individual=True,
    )

    # ② ▼ NEW: per-head HV CosSim 그래프
    hv_save = save_path.replace(".png", "_hv.png") if save_path else None
    if (nm.get("hv_cos_mean") is not None and nm["hv_cos_mean"].size > 0 and
        mm.get("hv_cos_mean") is not None and mm["hv_cos_mean"].size > 0):
        visualize_head_av_cosine(
            cos_normal   = nm["hv_cos_mean"],   # (n_steps, H)
            cos_mem      = mm["hv_cos_mean"],   # (n_steps, H)
            normal_label = "Normal Prompts",
            mem_label    = "Memorized Prompts",
            save_path    = hv_save,
        )
    else:
        print("⚠️ HV cosine 데이터 없음 — _ValueAwareAttnProcessor 가 설치됐는지 확인")

    # ③ 콘솔 요약
    print("\n" + "=" * 70)
    print(f"{'지표':<22} {'Normal':>18} {'Memorized':>18} {'Δ(M-N)':>10}")
    print("=" * 70)
    for key, label in [
        ("ce_s2p_mean",  "CE(s→p)"),
        ("ce_p2s_mean",  "CE(p→s)"),
        ("h_s_mean",     "H(special)"),
        ("h_p_mean",     "H(prompt)"),
        ("diff_mean",    "ΔCE"),
        ("cos_sim_mean", "CosSim(attn)"),
        ("ratio_mean",   "Ratio(EOS/P)"),
    ]:
        n_val = np.nanmean(nm[key]); m_val = np.nanmean(mm[key])
        print(f"{label:<22} {n_val:>18.4f} {m_val:>18.4f} {m_val-n_val:>+10.4f}")

    # HV 헤드별 요약
    if nm.get("hv_cos_mean") is not None and nm["hv_cos_mean"].size > 0:
        print("-" * 70)
        print("  HV CosSim  (mean over steps, per head):")
        for h in range(nm["hv_cos_mean"].shape[1]):
            nv = np.nanmean(nm["hv_cos_mean"][:, h])
            mv = np.nanmean(mm["hv_cos_mean"][:, h])
            marker = " ◀ Δ max" if abs(mv - nv) == max(
                abs(np.nanmean(mm["hv_cos_mean"][:, hh]) -
                    np.nanmean(nm["hv_cos_mean"][:, hh]))
                for hh in range(nm["hv_cos_mean"].shape[1])
            ) else ""
            print(f"    Head {h:2d}:  Normal={nv:+.4f}  Mem={mv:+.4f}"
                  f"  Δ={mv-nv:+.4f}{marker}")
    print("=" * 70)

    return nm, mm


# 클래스에 바인딩
SDAttentionVisualizer.compare_prompt_lists_hv = compare_prompt_lists_hv

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    # INSERT_YOUR_CODE
    import csv

    # CSV 파일에서 NORMAL_PROMPTS 리스트 불러오기
    NORMAL_PROMPTS = []
    import csv

    with open("/nas/home/jiyoon/vast/cvpr/unmemorized_laion_prompts.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            cap = row.get("Caption")
            if cap:
                NORMAL_PROMPTS.append(cap)
    # NORMAL_PROMPTS = [
    #     "a photo of an astronaut riding a horse on the moon",
    #     "a painting of a cat sitting on a red couch",
    #     "a beautiful sunset over the ocean",
    #     "a dog playing in the park on a sunny day",
    # ]
    

    MEM_PROMPTS = [
        "The No Limits Business Woman Podcast",
        "Full body U-Zip main opening - Full body U-Zip main opening on front of bag for easy unloading when you get to camp",
        "Mothers influence on her young hippo",
        "\"Watch: Passion Pit's New Video, \"\"Lifted Up (1985)\"\"\"",
        "Talks on the Precepts and Buddhist Ethics",
        "Sony Won't Release <i>The Interview</i> on VOD",
        "<i>I Am Chris Farley</i> Documentary Releases First Trailer",
        "<em>South Park: The Stick of Truth</em> Review (Multi-Platform)",
        "<i>X-Men: Days of Future Past</i> Director Joins Vince Gilligan's <i>Battle Creek</i> Pilot",
        "Insights with Laura Powers",
        "As Punisher Joins <i>Daredevil</i> Season Two, Who Will the New Villain Be?",
        "Aretha Franklin Files $10 Million Suit Over Patti LaBelle Fight Story On Satire Website",
        "Hawkgirl Cast in <i>Arrow</i>/<i>Flash</i> Spinoff Series For The CW",
        "Here's What You Need to Know About St. Vincent's Apple Music Radio Show",
        "Daniel Radcliffe Dons a Beard and Saggy Jeans in Trailer for BBC GTA Miniseries <i>The Gamechangers</i>",
        "The Happy Scientist",
        "The Health Mastery Caf\u00e9 with Dr. Dave",
        "\"\"\"Listen to The Dead Weather's New Song, \"\"\"\"Buzzkill(er)\"\"\"\"\"\"\"",
        "There's a <i>Mrs. Doubtfire</i> Sequel in the Works",
        "DC All Stars podcast",
        "35 Possible Titles for the <i>Mrs. Doubtfire</i> Sequel",
        "\"Listen to The Dead Weather's New Song, \"\"Buzzkill(er)\"\"\"",
        "Rambo 5 und Rocky Spin-Off - Sylvester Stallone gibt Updates",
        "Anna Kendrick is Writing a Collection of Funny, Personal Essays",
        "Breaking Down the $12 in Your Six-Pack of Craft Beer",
        "<i>It's Always Sunny</i> Gang Will Turn Your Life Around with Self-Help Book",
        "Renegade RSS Laptop Backpack - View 91",
        "Passion. Podcast. Profit.",
        "Prince Reunites With Warner Brothers, Plans New Album",
        "Here's Who Ian McShane May Be Playing in <i>Game of Thrones</i> Season Six",
        "Future Steve Carell Movie Set In North Korea Canceled By New Regency",
        "Gary Ryan Moving Beyond Being Good\u00ae",
        "Will Ferrell, John C. Reilly in Talks for <i>Border Guards</i>",
        "25 Timeless <i>Golden Girls</i> Memes and Quotables",
        "Long-Lost F. Scott Fitzgerald Story Rediscovered and Published, 76 Years Later",
        "Watch the First Episode of <i>Garfunkel and Oates</i>",
        "J Dilla's Synthesizers, Equipment Donated to Smithsonian Museum",
        "Foyer painted in HABANERO",
        "Living in the Light with Ann Graham Lotz",
        "<i>The Colbert Report</i> Gets End Date",
        "Brit Marling-Zal Batmanglij Drama Series <i>The OA</i> Gets Picked Up By Netflix",
        "Chris Messina In Talks to Star Alongside Ben Affleck in <i>Live By Night</i>",
        "Netflix Strikes Deal with AT&T for Faster Streaming",
        "Foyer painted in WHITE",
        "Sound Advice with John W Doyle",
        "<i>The Long Dark</i> Gets First Trailer, Steam Early Access",
        "Watch the Trailer for NBC's <i>Constantine</i>",
        "Donna Tartt's <i>The Goldfinch</i> Scores Film Adaptation",
        "Air Conditioners & Parts",
        "If Barbie Were The Face of The World's Most Famous Paintings",
        "Designart Canada White Stained Glass Floral Design 29-in Round Metal Wall Art",
        "Aaron Paul to Play Luke Skywalker at LACMA Reading of <i>The Empire Strikes Back</i>",
        "Foyer painted in KHAKI",
        "Renegade RSS Laptop Backpack - View 31",
        "Renegade RSS Laptop Backpack - View 21",
        "Falmouth Navy Blue Area Rug by Andover Mills",
        "Emma Watson to play Belle in Disney's <i>Beauty and the Beast</i>",
        "George R.R. Martin Donates $10,000 to Wolf Sanctuary for a 13-Year-Old Fan",
        "Director Danny Boyle Is Headed To TV With FX Deal",
        "Plymouth Curtain Panel featuring Madelyn - White Botanical Floral Large Scale by heatherdutton",
        "Plymouth Curtain Panel featuring diamond_x_blue_gray by boxwood_press",
        "Axle Laptop Backpack - View 81",
        "Renegade RSS Laptop Backpack - View 3",
        "Anzell Blue/Gray Area Rug by Andover Mills",
        "Aero 51-204710 51 Series 15x10 Wheel, Spun, 5 on 4-3/4 BP, 1 Inch BS",
        "Melrose Gray/Blue Area Rug by Andover Mills",
        "Aero 58-905055PUR 58 Series 15x10 Wheel, SP, 5 on 5 Inch, 5-1/2 BS",
        "Lilah Gray Area Rug by Andover Mills",
        "George R.R. Martin to Focus on Writing Next Book, World Rejoices",
        "Baby Shower Turned Meteor Shower: Anne Hathaway Fights Off Aliens in Sci-Fi Comedy <i>The Shower</i>",
        "Ava DuVernay Won't Direct <i>Black Panther</i> After All",
        "Renegade RSS Laptop Backpack - View 51",
        "Shaw Floors Couture' Collection Ultimate Expression 15\u2032 Sahara 00205_19829",
        "Foyer painted in HIGH TIDE",
        "33 Screenshots of Musicians in Videogames",
        "<i>Breaking Bad</i> Fans Get a Chance to Call Saul with Albuquerque Billboard",
        "Shaw Floors Sandy Hollow III 15\u2032 Adobe 00108_Q4278",
        "Designart Blue Fractal Abstract Illustration Abstract Canvas Wall Art - 7 Panels",
        "Shaw Floors Couture' Collection Ultimate Expression 15\u2032 Peanut Brittle 00702_19829",
        "Aero 50-975035BLU 50 Series 15x7 Inch Wheel, 5 on 5 Inch BP 3-1/2 BS",
        "Aero 55-004220 55 Series 15x10 Wheel, 4-lug, 4 on 4-1/4 BP, 2 Inch BS",
        "Design Art Light in Dense Fall Forest with Fog Ultra Vibrant Landscape Oversized Circle Wall Art",
        "Renegade RSS Laptop Backpack - View 5",
        "Lilah Dark Gray Area Rug by Andover Mills",
        "FUSE Backpack 25 - View 81",
        "Obadiah Hand-Woven Wool Gray Area Rug by Mercury Row",
        "Foyer painted in SALTY AIR",
        "Shaw Floors Shaw Design Center Different Times II 12 Classic Buff 00108_5C494",
        "Shaw Floors Sandy Hollow Classic Iv 12\u2032 Sahara 00205_E0554",
        "Shaw Floors Value Collections All Star Weekend III 15\u2032 Net Desert Sunrise 00721_E0816",
        "Aero 50-284730 50 Series 15x8 Inch Wheel, 5 on 4-3/4 BP, 3 Inch BS",
        "Sarah Silverman Will Star in HBO Pilot from <i>Secret Diary of a Call Girl</i> Creator",
        "Shaw Floors Caress By Shaw Cashmere II Icelandic 00100_CCS02",
        "Freddy Adu Signs For Yet Another Club You Probably Don't Know",
        "Shaw Floors Nfa/Apg Color Express II Suitable 00712_NA209",
        "Shaw Floors Simply The Best Without Limits II Net Sandbank 00103_5E508",
        "Shaw Floors Shaw Flooring Gallery Burtonville Luminary 00201_5293G",
        "Shaw Floors Couture' Collection Ultimate Expression 12\u2032 Tropic Vine 00304_19698",
        "Shaw Floors Value Collections Cashmere I Lg Net Harvest Moon 00126_CC47B",
        "Shaw Floors Value Collections Passageway 2 12 Camel 00204_E9153",
        "Shaw Floors Shaw Design Center Different Times II 12 Silk 00104_5C494",
        "Aero 30-984540BLK 30 Series 13x8 Inch Wheel, 4 on 4-1/2 BP 4 Inch BS",
        "Shaw Floors Simply The Best Of Course We Can III 12\u2032 Sepia 00105_E9425",
        "Signature Purple Ombre Sugar Skull and Rose Bedding",
        "Aero 52-985030GOL 52 Series 15x8 Inch Wheel, 5 on 5 BP, 3 Inch BS IMCA",
        "Shaw Floors Shaw Flooring Gallery Ellendale 15\u2032 Ink Spot 00501_5301G",
        "Renegade RSS Laptop Backpack - View 41",
        "Pencil pleat curtains in collection Panama Cotton, fabric: 702-34",
        "Aero 30-974230BLK 30 Series 13x7 Inch Wheel, 4 on 4-1/4 BP 3 Inch BS",
        "Shaw Floors Value Collections Because We Can II 15\u2032 Net Sea Shell 00100_E9315",
        "Designart Pink Fractal Pattern With Swirls Abstract Wall Art Canvas - 6 Panels",
        "Read a Previously Unpublished F. Scott Fitzgerald Story",
        "Design Art Beautiful View of Paris Paris Eiffel Towerunder Red Sky Ultra Glossy Cityscape Circle Wall Art",
        "Shaw Floors Queen Point Guard 12\u2032 Flax Seed 00103_Q4855",
        "Shaw Floors Queen Sandy Hollow I 15\u2032 Peanut Brittle 00702_Q4274",
        "Aero 56-084530 56 Series 15x8 Wheel, Spun, 5 on 4-1/2 BP, 3 Inch BS",
        "Aero 31-974510BLK 31 Series 13x7 Wheel, Spun Lite 4 on 4-1/2 BP 1 BS",
        "Shaw Floors Town Creek III Sea Mist 00400_52S32",
        "Shiflett Gray/Blue/White Area Rug by Andover Mills",
        "Aero 53-204720 53 Series 15x10 Wheel, BLock, 5 on 4-3/4 BP, 2 Inch BS",
        "Shaw Floors Shaw Design Center Royal Portrush III 12\u2032 Crumpet 00203_5C613",
        "Shaw Floors Shaw Design Center Park Manor 12\u2032 Cashew 00106_QC459",
        "Emma Watson Set to Star Alongside Tom Hanks in Film Adaptation of Dave Eggers' <i>The Circle</i>",
        "Shaw Floors Pure Waters 15 Clam Shell 00102_52H11",
        "Aero 31-984210BLK 31 Series 13x8 Wheel, Spun 4 on 4-1/4 BP 1 Inch BS",
        "Shaw Floors SFA Timeless Appeal I 15\u2032 Tundra 00708_Q4311",
        "Aero 50-924750ORG 50 Series 15x12 Wheel, 5 on 4-3/4 BP, 5 Inch BS",
        "Shaw Floors SFA Tuscan Valley Cashmere 00701_52E29",
        "Shaw Floors Shaw Design Center Moment Of Truth Acorn 00700_5C789",
        "Renegade RSS Laptop Backpack - View 2",
        "Shaw Floors Nfa/Apg Barracan Classic I True Blue 00423_NA074",
        "Shaw Floors Shaw Design Center Different Times III 12 Soft Copper 00600_5C496",
        "Shaw Floors All Star Weekend I 15\u2032 Castaway 00400_E0141",
        "Shaw Floors Couture' Collection Ultimate Expression 15\u2032 Soft Shadow 00105_19829",
        "FUSE Backpack 25 - View 71",
        "Shaw Floors Caress By Shaw Cashmere I Lg Bismuth 00124_CC09B",
        "FUSE Backpack 25 - View 91",
        "Shaw Floors Value Collections Cozy Harbor I Net Waters Edge 00307_5E364",
        "Aero 31-904030RED 31 Series 13x10 Wheel, Spun Lite, 4 on 4 BP, 3 BS",
        "Shaw Floors Value Collections That's Right Net Sedona 00708_E0925",
        "Grieve Cream/Navy Area Rug by Bungalow Rose",
        "Shaw Floors Value Collections Explore With Me Twist Net Wave Pool 00410_E0849",
        "Shaw Floors SFA Shingle Creek Iv 15\u2032 Mojave 00301_EA519",
        "Shaw Floors Roll Special Xv540 Tropical Wave 00420_XV540",
        "Shaw Floors Value Collections Of Course We Can III 12\u2032 Net Shadow 00502_E9441",
        "Shaw Floors Value Collections Passageway 1 12 Net Classic Buff 00108_E9152",
        "Shaw Floors Caress By Shaw Cashmere Classic Iv Navajo 00703_CCS71",
        "Lilah Gray Area Rug By Andover Mills.",
        "Tremont Blue/Ivory Area Rug by Andover Mills",
        "Tab top lined curtains in collection Chenille, fabric: 702-22",
        "Shaw Floors Caress By Shaw Cashmere I Lg Heirloom 00122_CC09B",
        "Shaw Floors Shaw Design Center Sweet Valley II 12\u2032 Tuscany 00204_QC422",
        "Shaw Floors Shaw Design Center Sweet Valley III 12\u2032 Soft Shadow 00105_QC424",
        "Shaw Floors Sandy Hollow Classic III 12\u2032 Cashew 00106_E0552",
        "Aero 52-984740RED 52 Series 15x8 Wheel, 5 on 4-3/4 BP, 4 Inch BS IMCA",
        "Shaw Floors Shaw Design Center Sweet Valley II 15\u2032 Mountain Mist 00103_QC423",
        "Renegade RSS Laptop Backpack - View 4",
        "Shaw Floors Value Collections Cashmere I Lg Net Pebble Path 00722_CC47B",
        "Shaw Floors Couture' Collection Ultimate Expression 12\u2032 Almond Flake 00200_19698",
        "\"Daft Punk, Jay Z Collaborate on \"\"Computerized\"\"\"",
        "Meditation Floor Pillow",
        "Shaw Floors Value Collections Cashmere Iv Lg Net Navajo 00703_CC50B",
        "Netflix Hits 50 Million Subscribers",
        "Shaw Floors SFA Awesome 4 Linen 00104_E0741",
        "Shaw Floors SFA Enjoy The Moment I 15\u2032 Butterscotch 00201_0C138",
        "Foyer painted in BROWN BAG",
        "Shaw Floors Value Collections Of Course We Can I 15 Net Linen 00100_E9432",
        "Shaw Floors Value Collections Sandy Hollow Cl II Net Alpine Fern 00305_5E510",
        "Shaw Floors Caress By Shaw Cashmere Classic I Mesquite 00724_CCS68",
        "Shaw Floors Caress By Shaw Cashmere Classic I Rich Henna 00620_CCS68",
        "Lilah Teal Blue Area Rug by Andover Mills",
        "Pencil pleat curtain in collection Linen, fabric: 392-05",
        "Shaw Floors Foundations Elemental Mix II Pixels 00170_E9565",
        "Shaw Floors Nfa/Apg Detailed Artistry I Snowcap 00179_NA328",
        "Designart Circled Blue Psychedelic Texture Abstract Art On Canvas - 7 Panels",
        "Shaw Floors SFA My Inspiration I Textured Canvas 00150_EA559",
        "Peraza Hand-Tufted White Area Rug by Mercury Row",
        "Shaw Floors Caress By Shaw Cashmere Classic III Rich Henna 00620_CCS70",
        "Shaw Floors SFA Timeless Appeal III 12\u2032 Country Haze 00307_Q4314",
        "Pencil pleat curtain in collection Panama Cotton, fabric: 702-31",
        "Duhon Gray/Ivory Area Rug by Mercury Row",
        "Shaw Floors Value Collections Because We Can III 15\u2032 Net Birch Tree 00103_E9317",
        "Shaw Floors Value Collections Cashmere Iv Lg Net Jade 00323_CC50B",
        "Shaw Floors Shaw Floor Studio Bright Spirit II 15\u2032 Marzipan 00201_Q4651",
        "Pencil pleat curtains in collection Blackout, fabric: 269-12",
        "Shaw Floors Shaw Flooring Gallery Union City II 12\u2032 Golden Echoes 00202_5306G",
        "Green Fractal Lights In Fog Abstract Wall Art Canvas - 6 Panels",
        "Aero 52-984510BLK 52 Series 15x8 Wheel, 5 on 4-1/2 BP, 1 Inch BS IMCA",
        "3D Black & White Skull King Design Luggage Covers 007",
        "Shaw Floors Value Collections Take The Floor Twist II Net Biscotti 00131_5E070",
        "Aero 52-984720BLU 52 Series 15x8 Wheel, 5 on 4-3/4 BP, 2 Inch BS IMCA",
        "Bj\u00f6rk Explains Decision To Pull <i>Vulnicura</i> From Spotify",
        "\"Listen to Ricky Gervais Perform \"\"Slough\"\" as David Brent\"",
        "Brickhill Ivory Area Rug by Andover Mills",
        "Shaw Floors Caress By Shaw Cashmere Iv Bison 00707_CCS04",
        "Shaw Floors Value Collections What's Up Net Linen 00104_E0926",
        "Smithtown Latte Area Rug by Andover Mills",
        "Anderson Tuftex Natural State 1 Dream Dust 00220_ARK51",
        "Shaw Floors Caress By Shaw Cashmere I Lg Gentle Doe 00128_CC09B",
        "Shaw Floors Caress By Shaw Quiet Comfort Classic III Atlantic 00523_CCB98",
        "Shaw Floors Shaw Floor Studio Textured Story 15 Candied Truffle 55750_52B76",
        "Shaw Floors Resilient Residential Stone Works 720c Plus Glacier 00147_525SA",
        "Set of 6 Brother TZe-231 Black on White P-Touch Label",
        "Aero 31-984040GRN 31 Series 13x8 Wheel, Spun, 4 on 4 BP, 4 Inch BS",
        "Shaw Floors Anso Premier Dealer Great Effect I 15\u2032 Almond Flake 00200_Q4328",
        "Shaw Floors Spice It Up Tyler Taupe 00103_E9013",
        "Shaw Floors Shaw Flooring Gallery Inspired By II French Linen 00103_5560G",
        "Shaw Floors Caress By Shaw Quiet Comfort Classic III Froth 00520_CCB98",
        "Shaw Floors Value Collections All Star Weekend I 12 Net Crumpet 00203_E0792",
        "Pencil pleat curtains in collection Jupiter, fabric: 127-00",
        "Aero 30-904550RED 30 Series 13x10 Inch Wheel, 4 on 4-1/2 BP 5 Inch BS",
        "Shaw Floors Value Collections Solidify III 12 Net Natural Contour 00104_5E340",
        "Tab top curtains in collection Avinon, fabric: 131-15",
        "Shaw Floors Caress By Shaw Quiet Comfort Classic Iv Rich Henna 00620_CCB99",
        "Shaw Floors Value Collections Cashmere I Lg Net Barnboard 00525_CC47B",
        "Shaw Floors Caress By Shaw Cashmere Classic Iv Spruce 00321_CCS71",
        "Pencil pleat curtains in collection Jupiter, fabric: 127-50",
        "Shaw Floors Northern Parkway Crystal Clear 00410_52V34",
        "Shaw Floors Roll Special Xv863 Bare Mineral 00105_XV863",
        "Shaw Floors Leading Legacy Crystal Gray 00500_E0546",
        "Shaw Floors Roll Special Xy176 Sugar Cookie 00101_XY176",
        "FUSE Backpack 25 - View 61",
        "Aero 52-984520RED 52 Series 15x8 Wheel, 5 on 4-1/2 BP, 2 Inch BS IMCA",
        "Shaw Floors Roll Special Xv375 Royal Purple 00902_XV375",
        "Shaw Floors Caress By Shaw Cashmere III Lg Pacific 00524_CC11B",
        "Plymouth Curtain Panel featuring Christmas woofs by gkumardesign",
        "Designart Serene Maldives Seashore at Sunset Oversized Landscape Canvas Art - 4 Panels",
        "Shaw Floors Apd/Sdc Decordovan II 12\u2032 Country Haze 00307_QC392",
        "Shaw Floors Enduring Comfort I French Linen 00103_E0341",
        "Shaw Floors Value Collections Passageway 2 12 Pewter 00501_E9153",
        "Eugenia Brown Area Rug by Andover Mills",
        "Aero 53-904530BLU 53 Series 15x10 Wheel, BL, 5 on 4-1/2 BP 3 Inch BS",
        "Anderson Tuftex Anderson Hardwood Palo Duro Mixed Width Golden Ore 37212_AA777",
        "Shaw Floors Value Collections Passageway 2 12 Mocha Chip 00705_E9153",
        "Shaw Floors Caress By Shaw Cashmere III Lg Yearling 00107_CC11B",
        "Aero 51-980540GRN 51 Series 15x8 Wheel, Spun, 5 on WIDE 5, 4 Inch BS",
        "Annabel Green Area Rug by Bungalow Rose",
        "Melrose Gray/Yellow Area Rug by Andover Mills",
        "Shaw Floors Shaw Floor Studio Porto Veneri III 12\u2032 Golden Rod 00202_52U58",
        "Shaw Floors Shaw Design Center Sweet Valley I 15\u2032 Blue Suede 00400_QC421",
        "Shaw Floors Shaw Floor Studio Porto Veneri I 15\u2032 Cream 00101_52U55",
        "Shaw Floors Elemental Mix III Gentle Rain 00171_E9566",
        "Shaw Floors",
        "Shaw Floors Simply The Best Bandon Dunes Silver Leaf 00541_E0823",
        "Shaw Floors Caress By Shaw Quiet Comfort Classic I Deep Indigo 00424_CCB96",
        "Shaw Floors Value Collections Color Moxie Meteorite 00501_E9900",
        "Aero 51-985020GRN 51 Series 15x8 Wheel, Spun, 5 on 5 Inch, 2 Inch BS",
        "Shaw Floors That's Right Rustic Taupe 00706_E0812",
        "Shaw Floors Queen Thrive Fine Lace 00100_Q4207",
        "Shaw Floors Newbern Classic 15\u2032 Crimson 55803_E0950",
        "Shaw Floors Caress By Shaw Cashmere Classic II Spearmint 00320_CCS69",
        "Shaw Floors Value Collections Xvn05 (s) Soft Chamois 00103_E1236",
        "Ethelyn Lilah Area Rug by Andover Mills",
        "Shaw Floors Caress By Shaw Tranquil Waters Net Sky Washed 00400_5E062",
        "Shaw Floors SFA Find Your Comfort Tt I Lilac Field (t) 901T_EA817",
        "Shaw Floors Solidify III 15\u2032 Pewter 00701_5E267",
        "Red Exotic Fractal Pattern Abstract Art On Canvas-7 Panels",
        "Pencil pleat curtains 130 x 260 cm (51 x 102 inch) in collection Brooklyn, fabric: 137-79",
        "Shaw Floors All Star Weekend III Net Royal Purple 00902_E0773",
        "Shaw Floors Make It Yours (s) Dockside 00752_E0819",
        "Shaw Floors Caress By Shaw Cashmere III Lg Onyx 00528_CC11B",
        "Shaw Floors Value Collections Take The Floor Texture II Net Hickory 00711_5E067",
        "Pencil pleat curtains in collection Velvet, fabric: 704-15",
        "Shaw Floors Simply The Best Within Reach III Grey Fox 00504_5E261",
        "Shaw Floors SFA Vivid Colors I Moroccan Jewel 00803_0C160",
        "Pencil pleat curtains in collection Velvet, fabric: 704-18",
        "Shaw Floors Value Collections Xvn05 (s) Bridgewater Tan 00709_E1236",
        "Shaw Floors Shaw Flooring Gallery Highland Cove III 12 Sage Leaf 00302_5223G",
        "Shaw Floors Queen Harborfields II 15\u2032 Green Apple 00303_Q4721",
        "Shaw Floors Fusion Value 300 Canyon Shadow 00810_E0281"
        # ... memorized 프롬프트 추가
    ]
    # MEM_PROMPTS = MEM_PROMPTS[:10] + MEM_PROMPTS[-10:]

    viz = SDAttentionVisualizer(device="cuda")

    nm, mm = viz.compare_prompt_lists(
        normal_prompts=NORMAL_PROMPTS,
        mem_prompts=MEM_PROMPTS,
        num_inference_steps=50,
        seed=42,
        target_res=16,
        save_path="normal_vs_mem_comparison.png",
    )
    save_path = "normal_vs_mem_comparison.png"
    # ✅ 추가: ratio 전용 그래프
    ratio_save = save_path.replace(".png", "_ratio.png") if save_path else None
    visualize_summary_prompt_ratio(
        nm, mm,
        normal_label="Normal Prompts",
        mem_label="Memorized Prompts",
        save_path=ratio_save,
        show_individual=True,
    )

    # ▼ HV 포함 버전으로 교체
    nm, mm = viz.compare_prompt_lists_hv(
        normal_prompts=NORMAL_PROMPTS,
        mem_prompts=MEM_PROMPTS,
        num_inference_steps=50,
        seed=42,
        target_res=16,
        save_path="normal_vs_mem_comparison.png",
    )

