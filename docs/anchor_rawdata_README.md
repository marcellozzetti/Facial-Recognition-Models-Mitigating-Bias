# Raw-Data ANCHOR — patch isolado, não toca o pipeline F1–F5

> Anchor 🅓 — isola o efeito do **nosso preprocessing** (multi-face
> cleaning + MTCNN re-alignment) sobre F1/IR absolutos, mantendo o
> recipe da nossa baseline. **Pipeline paralelo: tudo em `data/raw_anchor/`**.
> Os Fatores 1–5 continuam intactos em `data/processed/fairface_aligned/`.

## 1. O que este anchor responde

Pergunta: *"Quanto da nossa F1=0.688 (CE+linear ResNet-50 @224) vem do
recipe vs. do nosso preprocessing?"*

| Cenário do resultado | Interpretação |
|---|---|
| F1 🅓 ≈ F1 controle (0.688) | preprocessing nosso é neutro em acurácia |
| F1 🅓 acima da F1 controle | nosso pre-processing **prejudica** acurácia (cleaning + MTCNN cortam sinal) |
| F1 🅓 abaixo da F1 controle | nosso pre-processing **ajuda** acurácia (fortalece o Fator 1) |

**Vale para Linha A?** Atribuição matched dos F1-F5 segue válida — todos
os 5 fatores compartilham o MESMO pipeline. 🅓 acrescenta uma camada de
*posicionamento absoluto*, não revisa atribuição.

## 2. Configuração — separação total dos F1-F5

| Aspecto | F1-F5 (pipeline atual) | Anchor 🅓 (NOVO) |
|---|---|---|
| CSV labels | `data/raw/fairface/fairface_labels_clean.csv` (72k) | `data/raw/fairface/fairface_labels.csv` (97k raw — já existente) |
| Imagens | `data/processed/fairface_aligned/` (MTCNN re-aligned) | `data/raw/bucket/{train,val}/*.jpg` (FairFace original, **read-only**) |
| Configs | `configs/experiments_*/`, `configs/dataset_factor/`, etc. | `configs/anchor_rawdata/` |
| Outputs | `outputs/definitive/{factor3,factor4,factor5,baseline}/` | `outputs/definitive/anchor_rawdata/` |
| Pipeline code | inalterado | inalterado (`dataset.py` é path-agnostic — só lê do config) |

**Recipe idêntica ao controle (CE+linear definitivo)**: ResNet-50
ImageNet + Linear + AdamW lr=1e-3 + bs=128 + 224×224 + 25 ép + critério
val_f1_macro + 3 seeds. **Único Δ controlado:** o dado.

## 3. Aquisição dos dados raw — **JÁ EM PLACE**

Os dados originais do FairFace já estão no disco em
`data/raw/bucket/` (97 698 arquivos, estrutura `train/N.jpg` + `val/N.jpg`,
verificado por auditoria). **Read-only** — o pipeline do anchor apenas
LÊ deste path; nenhuma modificação. Não há etapa de download necessária.

Integridade verificada: o CSV `data/raw/fairface/fairface_labels.csv`
(97 698 linhas) mapeia-se 100 % para arquivos existentes no
`data/raw/bucket/` (sample de 20 amostras: 20/20 OK).

## 4. Como rodar (depois que os dados estiverem em place)

```powershell
# Gerar configs (apenas a 1ª vez ou após editar generator)
python scripts/generate_anchor_rawdata_configs.py

# Sanity 1 seed × 2 ép (~5-10 min em RN-50 @224)
python scripts/run_all_experiments.py `
  --configs-dir configs/anchor_rawdata `
  --output-dir outputs/anchor_rawdata_sanity `
  --epochs 2 --only exp_anc_rawdata_s42 --device cuda

# Batch 3 seeds × 25 ép (~1h30 total, RN-50 leve @224)
python scripts/run_all_experiments.py `
  --configs-dir configs/anchor_rawdata `
  --output-dir outputs/definitive/anchor_rawdata `
  --epochs 25 --device cuda
```

## 5. Extensão do recompute (pós-execução)

Adicionar grupo no `scripts/recompute_checkpoint_criterion.py`:
```python
("Raw-data anchor 🅓", "definitive/anchor_rawdata", "exp_anc_rawdata_"),
```

## 6. Limitações declaradas

- Isola apenas **CSV (cleaning) + imagens (MTCNN)**. NÃO isola o
  protocolo de split (mantemos 80/10/10 estratificado, não o split
  oficial train/val do FairFace). Para testar isso adicionalmente:
  novo anchor (🅔), fora do escopo desta peça.
- A `padding=0.25` é a escolha default acima; alternativa `padding=1.25`
  daria outro anchor (variação). Decidir antes do download.
- O CSV do FairFace original usa coluna `file` com prefixo `train/` ou
  `val/`. Nossa `dataset.py` lida com isso via `os.path.join`
  (já é path-agnostic) — sem mudança de código.
