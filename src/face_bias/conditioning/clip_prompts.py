"""Config D — FiLM sobre embedding CLIP-text (512-dim).

Cap. 4, §4.7 (Configuração D). Alternativa moderna com sinal condicionante
semântico em vez do vetor MST discreto.

Estratégia de prompt ensembling (contra sensibilidade do CLIP a formulação
única — vide FairerCLIP, Dehdashtian 2024):

    prompt_k = "a photo of a person with Monk Skin Tone {k}"  para k ∈ {1,...,10}
    embedding = Σ_k softmax_k(SkinToneNet(image)) · CLIP_text(prompt_k)

O embedding resultante é injetado no FiLM (canal Config D).

Interface pública prevista:
    class CLIPPromptEnsembler:
        def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch16")
        def build_prompt_bank(self) -> torch.Tensor  # (10, 512)
        def encode(mst_softmax: torch.Tensor) -> torch.Tensor  # (N, 512)

TODO Etapa 3 (Jan-Mar/2027):
    [ ] Integrar CLIP text encoder (open_clip ou transformers)
    [ ] Cache do prompt bank (10 embeddings fixos)
    [ ] Weighted sum por MST softmax
    [ ] Smoke test: shape (N, 512), consistência
    [ ] Comparação de custo vs Config B (MST 10-dim direto)

Ver também:
    - src/face_bias/conditioning/film.py — FiLM base (mesma inserção)
    - Cap 2, §2.10 (justificativa das 8 alternativas)
"""

from __future__ import annotations

# TODO: implementação
