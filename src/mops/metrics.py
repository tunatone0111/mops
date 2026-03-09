"""SSCD 및 CLIP Score 메트릭 계산."""

from __future__ import annotations

import logging
import os

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# SSCD 모델 weights URL (facebook/sscd-copy-detection 공식 릴리스)
SSCD_DISC_LARGE_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt"


class SSCDEncoder:
    """SSCD (Self-Supervised Copy Detection) 기반 이미지 유사도 계산."""

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = device
        cache_dir = torch.hub.get_dir()
        model_path = os.path.join(cache_dir, "sscd_disc_large.torchscript.pt")
        if not os.path.exists(model_path):
            logger.info(f"SSCD 모델 다운로드: {SSCD_DISC_LARGE_URL}")
            torch.hub.download_url_to_file(SSCD_DISC_LARGE_URL, model_path)
        self.model = torch.jit.load(model_path, map_location=device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((288, 288)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """이미지를 SSCD feature vector로 인코딩."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return F.normalize(features, dim=-1)


class CLIPScorer:
    """Open CLIP (ViT-g-14) 기반 text-image alignment 점수 계산."""

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-g-14", pretrained="laion2b_s12b_b42k"
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-g-14")

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """이미지를 CLIP feature vector로 인코딩."""
        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model.encode_image(image_tensor)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """텍스트를 CLIP feature vector로 인코딩."""
        text_tokens = self.tokenizer([text]).to(self.device)
        features = self.model.encode_text(text_tokens)
        return F.normalize(features, dim=-1)

    def score_from_features(self, image_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """사전 인코딩된 feature 간 cosine similarity."""
        return (image_features @ text_features.T).squeeze().item()


def compute_all_metrics(
    image_paths: dict[int, dict],
    prompt_groups: list[dict],
    device: torch.device | str = "cuda",
) -> pd.DataFrame:
    """
    모든 이미지 쌍에 대해 SSCD 및 CLIP score 계산.
    원본 이미지 feature는 프롬프트 그룹당 한 번만 인코딩.
    """
    logger.info("SSCD 모델 로딩...")
    sscd = SSCDEncoder(device=device)

    logger.info("CLIP 모델 (ViT-g-14) 로딩...")
    clip_scorer = CLIPScorer(device=device)

    rows = []

    for prompt_idx, group in enumerate(prompt_groups):
        original_prompt = group["original"]
        paraphrases = group["paraphrases"]
        paths = image_paths[prompt_idx]

        # 원본 이미지 feature를 한 번만 인코딩
        with Image.open(paths["original"]) as original_img:
            original_sscd_feat = sscd.encode(original_img)
            original_clip_feat = clip_scorer.encode_image(original_img)

        original_text_feat = clip_scorer.encode_text(original_prompt)
        clip_original = clip_scorer.score_from_features(original_clip_feat, original_text_feat)

        for para_idx, para_prompt in enumerate(paraphrases):
            with Image.open(paths["paraphrases"][para_idx]) as para_img:
                para_sscd_feat = sscd.encode(para_img)
                para_clip_feat = clip_scorer.encode_image(para_img)

            with Image.open(paths["mitigations"][para_idx]) as mit_img:
                mit_sscd_feat = sscd.encode(mit_img)
                mit_clip_feat = clip_scorer.encode_image(mit_img)

            with Image.open(paths["switchings"][para_idx]) as sw_img:
                sw_sscd_feat = sscd.encode(sw_img)
                sw_clip_feat = clip_scorer.encode_image(sw_img)

            # SSCD 유사도 (원본 feature 재사용)
            sscd_orig_vs_para = F.cosine_similarity(original_sscd_feat, para_sscd_feat, dim=-1).item()
            sscd_orig_vs_mit = F.cosine_similarity(original_sscd_feat, mit_sscd_feat, dim=-1).item()
            sscd_orig_vs_switching = F.cosine_similarity(original_sscd_feat, sw_sscd_feat, dim=-1).item()

            # CLIP score (모든 이미지를 original prompt 기준으로 측정)
            clip_para = clip_scorer.score_from_features(para_clip_feat, original_text_feat)
            clip_mit = clip_scorer.score_from_features(mit_clip_feat, original_text_feat)
            clip_switching = clip_scorer.score_from_features(sw_clip_feat, original_text_feat)

            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "paraphrase_idx": para_idx,
                    "original_prompt": original_prompt,
                    "paraphrase_prompt": para_prompt,
                    "sscd_original_vs_paraphrase": sscd_orig_vs_para,
                    "sscd_original_vs_mitigation": sscd_orig_vs_mit,
                    "sscd_original_vs_switching": sscd_orig_vs_switching,
                    "clip_score_original": clip_original,
                    "clip_score_paraphrase": clip_para,
                    "clip_score_mitigation": clip_mit,
                    "clip_score_switching": clip_switching,
                }
            )

        logger.info(f"[{prompt_idx + 1}/{len(prompt_groups)}] 메트릭 계산 완료: {original_prompt!r}")

    return pd.DataFrame(rows)
