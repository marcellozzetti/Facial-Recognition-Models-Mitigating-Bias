"""Auditoria fenotípica do FairFace via SkinToneNet — Etapa 2.

Cap. 4, §4.2 (Etapa 2). Aplica o classificador MST pré-treinado sobre
o FairFace validation set inteiro (~10.954 imagens) e produz o vetor
MST 10-dim por imagem, com raça verdadeira anexada.

Saída esperada:
    pandas.DataFrame com colunas:
        file, race_true (7 classes), mst_argmax (1..10), mst_softmax (list[10])

Interface pública prevista:
    def audit_fairface(
        skintonenet: SkinToneNetInference,
        fairface_val_dir: Path,
        annotations: Path,
    ) -> pd.DataFrame

TODO Etapa 2 (Dez/2026):
    [ ] Loader FairFace val (train_val_labels_face_only.csv)
    [ ] Inferência batched com progress bar
    [ ] Salvar resultado em outputs/etapa2/audit_fairface_mst.parquet
    [ ] Smoke test com subset 500 imagens

Ver também:
    - src/face_bias/audit/cross_matrix.py — consumidor (matriz Contribuição 2)
    - src/face_bias/mst/skintonenet.py — provedor da inferência
"""

from __future__ import annotations

# TODO: implementação
