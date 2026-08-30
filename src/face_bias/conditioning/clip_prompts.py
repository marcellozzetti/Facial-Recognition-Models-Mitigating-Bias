"""Configuração C — FiLM sobre embedding CLIP-text (512-dim).

Cap. 4 §4.7. Alternativa onde o sinal condicionante é um embedding
semântico produzido por CLIP-text em vez do vetor MST discreto.

Estratégia de prompt ensembling (contra sensibilidade do CLIP a
formulação única — cf. FairerCLIP, Dehdashtian 2024):

    prompt_k = "a photo of a person with Monk Skin Tone {k}"  para k ∈ {1,...,10}
    embedding = Σ_k softmax_k(MSTClassifier(image)) · CLIP_text(prompt_k)

Dependência do ``transformers`` é lazy: só é carregado se a Config C
for de fato usada. Cache do prompt bank em disco evita recomputar
os 10 embeddings a cada execução.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CLIP_EMBED_DIM = 512
DEFAULT_TEMPLATE = "a photo of a person with Monk Skin Tone {k}"
DEFAULT_MODEL = "openai/clip-vit-base-patch16"


def build_prompts(template: str = DEFAULT_TEMPLATE) -> list[str]:
    return [template.format(k=k) for k in range(1, 11)]


class CLIPPromptEnsembler:
    """Codifica um softmax MST em um embedding CLIP-text ponderado.

    Uso:
        ensembler = CLIPPromptEnsembler()
        ensembler.build_prompt_bank()
        emb = ensembler.encode(mst_softmax)  # (N, 512)

    O prompt bank é uma matriz fixa (10, 512) computada uma vez.
    A codificação é apenas uma multiplicação matricial (softmax @ bank).
    """

    def __init__(
        self,
        clip_model_name: str = DEFAULT_MODEL,
        template: str = DEFAULT_TEMPLATE,
        cache_path: Optional[Path] = None,
        device: str | torch.device = "cpu",
    ):
        self.model_name = clip_model_name
        self.template = template
        self.cache_path = Path(cache_path) if cache_path else None
        self.device = torch.device(device)
        self.prompts = build_prompts(template)
        self._prompt_bank: Optional[torch.Tensor] = None

    def _load_cached_bank(self) -> Optional[torch.Tensor]:
        if self.cache_path is None or not self.cache_path.exists():
            return None
        arr = np.load(self.cache_path)
        if arr.shape != (10, CLIP_EMBED_DIM):
            logger.warning("Cache CLIP com shape inesperada %s, ignorando.", arr.shape)
            return None
        return torch.from_numpy(arr).float()

    def _save_bank(self, bank: torch.Tensor) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, bank.cpu().numpy())

    def build_prompt_bank(self, force: bool = False) -> torch.Tensor:
        """Constrói (ou carrega do cache) a matriz (10, 512) de embeddings."""
        if not force:
            cached = self._load_cached_bank()
            if cached is not None:
                self._prompt_bank = cached.to(self.device)
                return self._prompt_bank

        try:
            from transformers import CLIPModel, CLIPTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Config C requer `pip install transformers`. "
                "Alternativa: use `set_prompt_bank_external(tensor)` "
                "para injetar embeddings pré-computados."
            ) from e

        model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        with torch.inference_mode():
            tokens = tokenizer(
                self.prompts, padding=True, return_tensors="pt"
            ).to(self.device)
            text_emb = model.get_text_features(**tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        self._prompt_bank = text_emb.detach().float()
        self._save_bank(self._prompt_bank)
        return self._prompt_bank

    def set_prompt_bank_external(self, bank: torch.Tensor) -> None:
        """Permite injetar embeddings pré-computados (útil para testes)."""
        if bank.shape != (10, CLIP_EMBED_DIM):
            raise ValueError(f"bank precisa ter shape (10, {CLIP_EMBED_DIM})")
        self._prompt_bank = bank.to(self.device).float()

    def encode(self, mst_softmax: torch.Tensor) -> torch.Tensor:
        """Codifica um softmax MST (N, 10) em embeddings CLIP (N, 512)."""
        if self._prompt_bank is None:
            raise RuntimeError("Chame build_prompt_bank() antes de encode().")
        if mst_softmax.dim() != 2 or mst_softmax.shape[1] != 10:
            raise ValueError(f"esperado (N, 10); recebido {tuple(mst_softmax.shape)}")
        z = mst_softmax.to(self.device).float() @ self._prompt_bank
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    @property
    def embed_dim(self) -> int:
        return CLIP_EMBED_DIM
