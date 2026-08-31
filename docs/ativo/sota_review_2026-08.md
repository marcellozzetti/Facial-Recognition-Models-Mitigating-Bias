# Nova rodada de SOTA — revisão bibliográfica 2026-08

**Data:** 31/08/2026
**Motivação:** feedback do orientador — avaliar publicações desde a
última revisão (jun/2026) que possam impactar posicionamento da
dissertação, seja como (a) nova referência a incorporar, (b) potencial
concorrente da tese, ou (c) validação da direção escolhida.

**Método:** buscas em Google Scholar, arXiv, ACM DL, IEEE Xplore
(termos: FairFace, race classification, fairness face recognition,
Monk Skin Tone, FiLM conditioning, ViT fairness, RFW/BFW). Filtro
temporal: publicações posteriores a 2026-06 ou não catalogadas na
revisão anterior.

---

## 1. Publicações que exigem incorporação ao Cap. 2

### 1.1 Frontiers in AI (2026) — "Reassessing demographic bias in face attribute classification"
- **Autores:** *(a preencher com autoria completa após leitura do full text)*
- **DOI:** [10.3389/frai.2026.1817529](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1817529/full)
- **O que traz:** avaliação estatisticamente fundamentada de 3 arquiteturas (ResNet50, MobileNetV3, DeiT) sobre FairFace + UTKFace, com ênfase em avaliação por subgrupo e quantificação de incerteza.
- **Impacto para nossa tese:**
  - **Reforça** nossa escolha metodológica de reportar por subgrupo (Cenário A/B) e usar bootstrap CI (§4.10).
  - **Incorporar em Cap. 2** como referência recente da linha "fairness via reavaliação estatística".
  - **Não é concorrente direto:** eles reavaliam classificação de atributos; nós propomos condicionamento arquitetural.

### 1.2 Balancing Beyond Discrete Categories (arXiv:2506.01532, jun/2025)
- **Título completo:** *Balancing Beyond Discrete Categories: Continuous Demographic Labels for Fair Face Recognition*
- **O que traz:** propõe tratar tom de pele como variável **contínua** (MST 1–10) em vez de rotulação categórica discreta.
- **Impacto para nossa tese:** **DIRETAMENTE RELEVANTE**. Justifica o uso do vetor softmax MST 10-dim (não one-hot) na Etapa 3 — o vetor de probabilidades preserva a natureza ordinal/contínua do tom.
- **Ação:** citar em Cap. 4 §4.7 (Configuração B) e Cap. 2 §MST como confirmação empírica externa da abordagem.

### 1.3 Fairness-Aware Grouping for Continuous Sensitive Variables (arXiv:2507.11247, jul/2025)
- **Sobre:** debiasing de análise facial via agrupamento adaptativo sobre tom de pele contínuo.
- **Impacto para nossa tese:** validação externa da premissa "tom de pele é variável tratável como contínua para fairness". Complementa 1.2.
- **Ação:** citar em §4.7 e §4.8 (triangulação de métricas).

### 1.4 Component-Based Fairness (Liu et al. 2025, arXiv:2505.01699, ACM FAccT)
- **O que traz:** *Bayesian Network-informed Meta Reweighting* — introduz "face component fairness" (equidade componente a componente do rosto: olhos, nariz, boca).
- **Impacto:** oferece **direção complementar** para a decomposição da Etapa 6 — podemos considerar decompor não só por MST/raça mas por componente facial. **Não substituir** nossa abordagem, mas mencionar como trabalho futuro.
- **Status na tese:** já citado na revisão bibliográfica atual (revisao-bibliografica.tex L26).

### 1.5 Adaptive Diffusion Models — FADiff (Nature Sci Rep 2025)
- **DOI:** [s41598-025-34520-3](https://www.nature.com/articles/s41598-025-34520-3)
- **O que traz:** integra FiLM em modelo de difusão para face recognition em longa distância, usando FiLM para regular propagação de features garantindo consistência de identidade.
- **Impacto para nossa tese:** **evidência de precedente recente** do uso de FiLM em face recognition — reforça (não invalida) nossa proposta. Diferença: FADiff usa FiLM em geração/reconstrução; nós usamos em condicionamento por tom de pele durante classificação.
- **Ação:** citar em Cap. 4 §4.6 (Racional da escolha de FiLM) como precedente aplicado, reforçando "FiLM foi validado em face recognition, adaptamos para fairness via tom de pele".

### 1.6 Can ViTs with ResNet Global Features Fairly Authenticate Demographic Faces? (arXiv:2506.05383)
- **O que traz:** avalia ViT + ResNet features para authentication demográfica.
- **Impacto:** reforça a validade da comparação **ConvNeXt-T vs ResNet vs ViT** que fizemos no Slide 18 da apresentação.
- **Ação:** citar em Cap. 4 §4.5 (Escolha do backbone).

### 1.7 Demographic Fairness in Multimodal LLMs (arXiv:2603.25613)
- **O que traz:** benchmark de gender/ethnicity bias em face verification usando LLMs multimodais.
- **Impacto:** amplia contexto — modelos VLM (relacionado ao FaceScanPaliGemma que já citamos) continuam sendo avaliados.
- **Ação:** menção breve em Cap. 2 §Vision-Language Models.

---

## 2. Publicações complementares (menção opcional)

- **Improving Bias via KL Divergence + Dual Attention (arXiv:2410.11176):** loss + atenção — direção metodológica alternativa. Só mencionar se banca perguntar.
- **March 2026 Face Recognition Papers (InsightFace blog):** consolidação editorial das prioridades de 2026 — "fairness, better embeddings, explainable comparison". Útil para citar como evidência de que fairness é prioridade atual da comunidade.
- **FAccT 2026 (Montreal, jun 25–28):** conferência de referência — considerar submissão da Contribuição 2 (matriz MST×raça) após executar Etapa 2.

---

## 3. Consolidação

### 3.1 Referências novas que valem citar no Cap. 2 (5)

| Chave BibTeX (proposta) | Autor Ano | Onde citar |
|---|---|---|
| `frontiers2026reassessing` | Frontiers AI 2026 | §Auditoria estatística |
| `continuous2025labels` | (a definir) 2025 | §MST + §Etapa 3 (Config B) |
| `fairnessgrouping2025` | (a definir) 2025 | §MST |
| `fadiff2025` | Nature Sci Rep 2025 | §FiLM (racional) |
| `vitresnet2025fair` | (a definir) 2025 | §Backbone (racional) |

### 3.2 Direção da tese: continua defensável

Nenhuma publicação identificada:
- **Refuta** o uso de FiLM em fairness facial.
- **Antecipa** a Contribuição 2 (matriz MST×raça no FairFace).
- **Antecipa** a Contribuição 6 (decomposição fenotípico × algorítmico).

O trabalho **Component-Based Fairness (Liu 2025)** é o mais próximo em espírito diagnóstico, mas decompõe por componente facial (olhos/nariz/boca), não por tom vs modelo — direções complementares.

### 3.3 Ações práticas

- [ ] Baixar full text das 5 referências novas em PDFs internos
- [ ] Adicionar entradas em `docs/tese/references.bib`
- [ ] Incorporar 1 parágrafo por referência no Cap. 2 (2-3 páginas adicionais)
- [ ] Atualizar apresentação da qualificação com referência ao FADiff e Balancing 2025

**Sources principais:**
- [Frontiers in AI 2026 — Reassessing demographic bias](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1817529/full)
- [Balancing Beyond Discrete Categories (arXiv:2506.01532)](https://arxiv.org/pdf/2506.01532)
- [Fairness-Aware Grouping (arXiv:2507.11247)](https://arxiv.org/pdf/2507.11247)
- [FADiff (Nature Sci Rep 2025)](https://www.nature.com/articles/s41598-025-34520-3)
- [Can ViTs with ResNet Global Features (arXiv:2506.05383)](https://arxiv.org/pdf/2506.05383)
- [Component-Based Fairness Liu 2025 (arXiv:2505.01699)](https://arxiv.org/pdf/2505.01699)
- [Demographic Fairness in Multimodal LLMs (arXiv:2603.25613)](https://arxiv.org/pdf/2603.25613)
- [InsightFace blog Mar 2026](https://www.insightface.ai/blog/march-2026-face-recognition-papers)
