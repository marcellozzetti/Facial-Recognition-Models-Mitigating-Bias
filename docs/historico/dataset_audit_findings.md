# Auditoria de Integridade do Dataset FairFace — Achados

**Data:** 2026-05-14 · **Script:** `scripts/audit_multi_face.py`
**Diretriz:** 4 (kickoff 2026-05-11 — "limpar imagens com mais de uma face")
**Dados brutos:** `outputs/audit/multi_face_audit_summary.md`,
`outputs/audit/multi_face_audit.csv`

---

## 1. O que foi testado

Rodamos o detector **MTCNN** sobre **todas as 97 698 imagens originais**
do FairFace e contamos **quantas faces** cada imagem contém.

**Esclarecimento conceitual (importante).** O FairFace tem **um único
rótulo por imagem** (uma raça, um gênero, uma idade). O problema **não**
é multi-rótulo — é **multi-face**: quando uma imagem contém mais de uma
pessoa, o rótulo único fica **ambíguo** (a qual rosto o "Black" se
refere?). Há ainda imagens em que o MTCNN **não detecta nenhuma face**
(linha inutilizável). Ambos comprometem a integridade do par
(imagem, rótulo) usado no treino.

---

## 2. O que encontramos — números

| Situação | Imagens | % do total |
|---|---:|---:|
| **1 face** (rótulo unívoco — "limpo") | **72 749** | **74,46%** |
| ≥ 2 faces (rótulo ambíguo — "multi") | 24 531 | 25,11% |
| 0 faces (não-detectável — "zero") | 418 | 0,43% |
| **Total** | **97 698** | 100% |

**1 em cada 4 imagens do FairFace tem rotulação ambígua.** Um modelo
treinado no dataset completo aprende com **~25% de supervisão ruidosa**.

Distribuição de `n_faces` (multi-face):

| n_faces | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| imagens | 18 675 | 4 240 | 1 042 | 341 | 124 | 55 | 28 | 26 |

Casos extremos: uma imagem rotulada "Black" (`train/5780.jpg`) tem
**20 faces detectadas**; outra "White" tem 18; "Latino_Hispanic", 15.
São claramente fotos de grupo/cena rotuladas como se fossem retrato
individual.

---

## 3. O achado central — viés de cena correlacionado com raça

A taxa de imagens multi-face **não é uniforme entre raças**:

| Raça | Multi-face | % da raça | avg n_faces |
|---|---:|---:|---:|
| **Black** | 4 134 | **29,98%** | 2,45 |
| Latino_Hispanic | 4 337 | 28,93% | 2,39 |
| Indian | 3 741 | 27,04% | 2,40 |
| Southeast Asian | 2 895 | 23,71% | 2,29 |
| Middle Eastern | 2 363 | 22,67% | 2,32 |
| White | 4 201 | 22,57% | 2,29 |
| **East Asian** | 2 860 | **20,67%** | 2,24 |

**Spread ≈ 9,3 pontos percentuais** (Black 30,0% vs East Asian 20,7%).
Imagens de certos grupos são, com mais frequência, **cenas com várias
pessoas** — não retratos individuais.

O viés do **detector** (zero-face) também é racialmente assimétrico,
em direção quase oposta:

| Raça | Zero-face | % da raça |
|---|---:|---:|
| White | 140 | **0,75%** |
| East Asian | 73 | 0,53% |
| Middle Eastern | 51 | 0,49% |
| Black | 63 | 0,46% |
| Southeast Asian | 34 | 0,28% |
| Indian | 32 | 0,23% |
| Latino_Hispanic | 25 | **0,17%** |

→ Tanto a **composição da cena** (quantas pessoas na foto) quanto a
**falha de detecção do MTCNN** carregam viés demográfico — *antes* de
qualquer modelo de classificação.

**Interpretação.** O FairFace foi construído para corrigir o
**desbalanceamento de rótulo** (mesmo nº de imagens por raça). Mas ele
**não corrige o viés de contexto/cena**: fotos de alguns grupos vêm
mais de contextos coletivos. É um viés de dataset **independente do
modelo** — achado próprio, candidato a paper curto (Linha C da tese).

---

## 4. Decisão e impacto no dataset (Opção A)

Adotada a **Opção A: manter apenas `n_faces == 1`** (rótulo unívoco).

| | Original (97k) | Limpo (n_faces=1) |
|---|---:|---:|
| White | 18 612 | 14 271 |
| Latino_Hispanic | 14 990 | 10 628 |
| East Asian | 13 837 | 10 904 |
| Indian | 13 835 | 10 062 |
| Black | 13 789 | 9 592 |
| Southeast Asian | 12 210 | 9 281 |
| **Middle Eastern** | 10 425 | **8 011** (nova minoria) |
| **Total** | **97 698** | **72 749** (−25,54%) |

Pós-filtro, o `undersampling` balanceia pela nova minoria (Middle
Eastern, 8 011) → dataset balanceado de **56 077 imagens** (7 × 8 011).
Artefato: `data/raw/fairface/fairface_labels_clean.csv`
(gerado por `scripts/filter_dataset_clean.py`, idempotente).

---

## 5. Como isso entra na tese

Dois usos distintos:

1. **Fator "Dataset" da decomposição causal** — o *efeito* de treinar
   no dataset limpo vs original é medido em
   [dataset_factor_results.md](dataset_factor_results.md) (protocolo
   3-seed). Resultado defensável: a limpeza contribui para **acurácia
   (CE +1,35pp)**, não para fairness.
2. **Achado de viés de cena (Linha C)** — o spread de 9,3 pp na taxa
   multi-face por raça é um **achado de dataset**, independente do
   modelo: o FairFace carrega viés contextual além do desbalanceamento
   de rótulo que ele corrige. Material para paper curto / capítulo.

---

## 6. Reprodução

```powershell
.\.venv\Scripts\python.exe scripts/audit_multi_face.py      # gera outputs/audit/
.\.venv\Scripts\python.exe scripts/filter_dataset_clean.py  # gera o CSV limpo
```
