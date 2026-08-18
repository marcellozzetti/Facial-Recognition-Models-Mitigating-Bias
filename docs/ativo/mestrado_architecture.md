# Arquitetura de código — pipeline mestrado

**Última atualização:** 2026-08-01
**Contexto:** qualificação agendada para 30/set/2026; orientador aprovou o início
dos próximos passos após a submissão formal.

Este documento é o **índice de navegação** da estrutura de código criada para as
6 Etapas do pipeline metodológico definido no Cap. 4 da monografia. Cada componente
listado abaixo tem um arquivo skeleton com docstring, referências ao capítulo/seção
correspondente da dissertação e um bloco `TODO` explícito com o prazo do cronograma.

## Mapa de correspondência Cap. 4 ↔ código

| Etapa (Cap. 4 §4.2) | Prazo | Módulo | Arquivos skeleton |
|---|---|---|---|
| **1** — Classificador MST | Nov/2026 | `src/face_bias/mst/` | `skintonenet.py`, `validation.py`, `sensitivity.py` |
| **2** — Auditoria fenotípica | Dez/2026 | `src/face_bias/audit/` | `fairface_mst.py`, `cross_matrix.py` |
| **3** — Classificador racial condicionado | Jan-Mar/2027 | `src/face_bias/conditioning/` | `film.py`, `clip_prompts.py`, `injector.py` |
| **4** — Comparação vs baselines | Abr/2027 | `src/face_bias/baselines/`, `src/face_bias/fairness/` | `fscl_plus.py`, `group_dro.py`, `fineface.py`, `adversarial_debias.py`, `disparity_ratio.py`, `worst_class_f1.py`, `equal_opportunity.py`, `equalized_odds.py`, `pareto.py` |
| **5** — Transferência fair | Mai/2027 | `src/face_bias/transfer/` | `rfw_eval.py`, `bfw_eval.py`, `bisenet_pixel_info.py`, `quality_descriptors.py` |
| **6** — Síntese decompositiva | Jun/2027 | `src/face_bias/decomposition/` | `error_decomp.py` |

## Estrutura do repositório

```
Facial-Recognition-Models-Mitigating-Bias/
│
├── src/face_bias/                    # Package principal
│   ├── mst/            🆕 Etapa 1     (3 skeleton files)
│   ├── audit/          🆕 Etapa 2     (2 skeleton files)
│   ├── conditioning/   🆕 Etapa 3     (3 skeleton files — FiLM + CLIP + injector)
│   ├── baselines/      🆕 Etapa 4     (4 skeleton files — 4 baselines faltantes)
│   ├── fairness/       🆕 Etapa 4     (5 skeleton files — triangulação + Pareto)
│   ├── transfer/       🆕 Etapa 5     (4 skeleton files — RFW/BFW + confounders)
│   ├── decomposition/  🆕 Etapa 6     (1 skeleton file)
│   ├── models/         ♻️ herança MBA (ConvNeXt-T já registrado!)
│   ├── training/       ♻️ herança MBA
│   ├── evaluation/     ♻️ herança MBA (base — integrar fairness/)
│   ├── data/           ♻️ herança MBA
│   ├── preprocessing/  ♻️ herança MBA (MTCNN)
│   ├── interpretability/ ♻️ herança MBA (GradCAM, t-SNE)
│   ├── cli/            ♻️ herança MBA (face-bias-*)
│   ├── config/         ♻️ herança MBA
│   └── utils/          ♻️ herança MBA
│
├── configs/mestrado/                 🆕 (14 arquivos)
│   ├── common/                       (seeds, hyperparams, datasets)
│   ├── stages/                       (9 configs: 1 por etapa + 3 ablations)
│   └── production.yaml               (orquestração root)
│
├── tests/mestrado/                   🆕 (7 arquivos + __init__)
│   ├── unit/                         (test_film, test_mst_wrapper, test_fairness_metrics, test_pareto)
│   ├── integration/                  (test_etapa3_ablation, test_etapa5_transfer)
│   └── smoke/                        (test_full_mestrado_pipeline)
│
├── pipelines/                        (herança MBA + 6 novos)
│   ├── 01_download_dataset.py        ♻️ herança
│   ├── 02_preprocess.py              ♻️ herança
│   ├── 03_mst_inference.py           🆕 Etapa 1
│   ├── 04_fairface_audit.py          🆕 Etapa 2
│   ├── 05_train_ablation.py          🆕 Etapa 3
│   ├── 06_evaluate_baselines.py      🆕 Etapa 4
│   ├── 07_transfer_downstream.py     🆕 Etapa 5
│   └── 08_decomposition.py           🆕 Etapa 6
│
├── docs/
│   ├── tese/                         ♻️ monografia LaTeX (imutável até defesa)
│   ├── ativo/                        (materiais de trabalho, incluindo este doc)
│   └── historico/                    (arquivo — reuniões, MBA, security audit)
│       ├── mba_experiments/          🆕 configs MBA arquivados (73 arquivos)
│       └── security_audit.md         ♻️ auditoria em 3 waves
│
├── tests/smoke/                      ♻️ herança MBA (7 smoke tests passando)
├── data/, outputs/, notebooks/, scripts/  ♻️ herança MBA
└── pyproject.toml, requirements.txt  ♻️ atualizado (Wave 3 security)
```

## Fase 0 — preparação (Ago-Out/2026, antes da qualificação)

**Prioridade:** manter foco na qualificação (30/set). Preparação de código é secundária.

| Semana | Tarefa | Módulo | Bloqueia? |
|---|---|---|---|
| Ago (semanas 1-2) | Estrutura criada (este commit) | — | ✅ done |
| Ago (semanas 3-4) | Baixar SkinToneNet weights + smoke test | `mst/skintonenet.py` | não bloqueia qual. |
| Set (semanas 1-3) | Finalizar apresentação + treinar apresentação | (foco: qualificação) | 🔒 prioridade máxima |
| Set (semana 4) | **QUALIFICAÇÃO 30/09** | — | 🎯 marco |
| Out (semanas 1-2) | Aplicar eventuais correções da banca no texto | `docs/tese/` | pode bloquear se banca pedir revisões |
| Out (semanas 3-4) | Iniciar validação humana interna (Etapa 1) | `mst/validation.py` | inicia Fase 1 |

## Ordem de implementação recomendada

Para cada Etapa, sugere-se a ordem:

1. **Unit tests primeiro** (TDD parcial) — clarifica contratos das funções
2. **Módulo `src/`** — implementação
3. **Pipeline em `pipelines/`** — script de orquestração
4. **Config em `configs/mestrado/stages/`** — hiperparâmetros
5. **Smoke test em `tests/mestrado/smoke/`** — validação end-to-end
6. **Documentação em `docs/ativo/`** — resultados e lições

## Checklist de disponibilidade de recursos externos

Antes de iniciar cada etapa, confirmar:

- [ ] **Etapa 1**: SkinToneNet weights públicos no arXiv 2603.02475 (Matias 2026)
- [ ] **Etapa 1**: acesso ao FairFace validation set completo (~10.954 imagens)
- [ ] **Etapa 3**: GPU disponível (ConvNeXt-T @ batch 64 exige ~8GB VRAM)
- [ ] **Etapa 4**: código público dos 6 baselines (Park, Sagawa, Manzoor, Zhang)
- [ ] **Etapa 5**: RFW e/ou BFW datasets (protocolos oficiais de pares)
- [ ] **Etapa 5**: BiSeNet weights públicos para pixel information
- [ ] **Etapa 6**: nenhum recurso externo além dos anteriores

## Rigor experimental (aplicável a todas as etapas)

Conforme Cap. 4 §4.10 (Rigor experimental) e feedback registrado em `memory`:

- **3 sementes independentes** por experimento (42, 1, 2)
- **Comparação pareada casada** entre configurações
- **IC 95% via bootstrap não paramétrico**
- **Reporte estratificado** por raça e por interseção race × gender

Estas convenções estão declaradas em `configs/mestrado/common/seeds.yaml`
e `configs/mestrado/common/hyperparams.yaml` — herdadas por todos os
configs `stages/etapa3_ablation_*.yaml` via `!include`.

## Convenções de nomenclatura

- **Etapas** referenciadas como `etapa1`, `etapa2`, ..., `etapa6` (lowercase, sem hífen)
- **Configurações de ablation** referenciadas como `A`, `B`, `C`, `D` (uppercase, sem hífen)
- **Sementes** sempre `42`, `1`, `2` (ordem fixa para logs)
- **Outputs** em `outputs/etapa{N}/[sub_id]/metrics_seed{S}.json`
- **Módulos novos** em `snake_case` (Python convention); **classes** em `PascalCase`

## Referências cruzadas

- `docs/tese/tex/textual/metodologia.tex` — Cap. 4 (metodologia formal)
- `docs/tese/tex/cronograma.tex` — Cap. 5 (cronograma e riscos)
- `docs/historico/security_audit.md` — auditorias de dependências (Waves 1/2/3)
- `docs/ativo/04_pesquisa_bibliografica/` — 104 fichas do corpus revisado
- `REVIEW_AND_PLAN.md` — histórico do refactor MBA → mestrado (2026-05)

## Próximos documentos a criar (por Etapa)

À medida que cada etapa avança, criar em `docs/ativo/`:

- `etapa1_report.md` — protocolo humano + resultados sensitivity
- `etapa2_report.md` — matriz MST × raça publicada (Contribuição 2)
- `etapa3_report.md` — ablation A/B/C (baseline, FiLM+MST, FiLM+CLIP) com IC 95% e ranks
- `etapa4_report.md` — comparativo baselines + Pareto
- `etapa5_report.md` — transferência fair RFW/BFW + confounders
- `etapa6_report.md` — decomposição final (Contribuição 6)

Estes documentos alimentarão os capítulos 5+ da dissertação final (a escrever pós-qualificação).
