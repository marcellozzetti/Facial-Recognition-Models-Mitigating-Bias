# Saneamento de citações — referências verificadas

> **Documento de saneamento crítico.** Em 2026-05-25, na reunião com o
> orientador, foi constatado que múltiplas citações nos documentos
> anteriores estavam com **autoria incorreta**. Verificação confirmou
> que 5 de 7 papers principais haviam sido atribuídos a autores
> errados (confabulação a partir de extrações superficiais via
> WebFetch). Este documento centraliza a verificação arXiv→autoridade
> de toda referência citada nos materiais ativos.
>
> **Política a partir de agora:** nenhum paper é citado em `docs/ativo/`
> sem entrada ✅ VERIFIED nesta tabela. Toda citação aponta para o
> registro abaixo.
>
> **Metodologia de verificação:** WebFetch sobre `https://arxiv.org/abs/<ID>`,
> DOI oficial, ou Google Scholar quando arXiv não disponível. Extração:
> título exato, lista completa de autores, data de submissão, venue
> declarado em comments. **Não há interpretação de conteúdo** —
> somente metadados.
>
> Versão consolidada: 2026-05-25 (27 papers verificados).

## Legenda

- ✅ **VERIFIED** — metadados confirmados em fonte primária
- ⚠️ **CONTENT-TO-VERIFY** — autoria correta, mas descrição de conteúdo
  feita anteriormente sem leitura integral; usar apenas para citação
  bibliográfica, não para descrição de método/resultados
- ⏳ **TO_VERIFY** — pendente

## Seção 1 — Papers SOTA / Fairness em classificação racial facial

| # | arXiv / DOI | Autoria CORRETA | Título exato | Venue / Ano | Status | Erro anterior |
|---|---|---|---|---|---|---|
| S1 | arXiv:2410.24148 | **Nouar AlDahoul, Myles Joshua Toledo Tan, Harishwar Reddy Kasireddy, Yasir Zaki** | Exploring Vision Language Models for Facial Attribute Recognition: Emotion, Race, Gender, and Age | arXiv preprint, 2024 (submetido 31 Oct 2024) | ✅ VERIFIED | "Hassanpour et al. 2024" — INCORRETO |
| S2 | arXiv:1908.04913 | **Kimmo Kärkkäinen, Jungseock Joo** | FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age | WACV 2021 (arXiv v1 14 Aug 2019) | ✅ VERIFIED | (correto) |
| S3 | arXiv:2408.16881 | **Ayesha Manzoor, Ajita Rattani** | FineFACE: Fair Facial Attribute Classification Leveraging Fine-grained Features | **ICPR 2024** (Springer LNCS) — arXiv v1 29 Aug 2024 | ✅ VERIFIED | "Liu et al. 2024" — INCORRETO; venue inicialmente registrado como "preprint" — corrigido após triagem (Semantic Scholar 2026-05-25) |
| S4 | arXiv:2404.09454 | **Sepehr Dehdashtian, Bashir Sadeghi, Vishnu Naresh Boddeti** | Utility-Fairness Trade-Offs and How to Find Them (U-FaTE) | IEEE/CVF CVPR 2024 | ✅ VERIFIED | "Sojitra et al. 2024" — INCORRETO |
| S5 | arXiv:2207.10888 | **Xiaofeng Lin, Seungbae Kim, Jungseock Joo** | FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification | ECCV 2022 | ✅ VERIFIED | (correto) |
| S6 | arXiv:2312.14626 | **Iris Dominguez-Catena, Daniel Paternain, Mikel Galar** | DSAP: Analyzing Bias Through Demographic Comparison of Datasets | Information Fusion, 2024 | ✅ VERIFIED | "Sánchez-Sánchez et al. 2024" — INCORRETO |
| S7 | arXiv:2504.08396 | **Valentin Lafargue, Emmanuelle Claeys, Jean-Michel Loubes** | Fairness is in the details: Face Dataset Auditing | ECML PKDD 2025 (LNCS vol 16022) | ✅ VERIFIED | "Galera-Zarco et al. 2025" — INCORRETO |
| S8 | arXiv:1812.00194 | **Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, Yaohai Huang** | Racial Faces in-the-Wild: Reducing Racial Bias by Information Maximization Adaptation Network (RFW) | **ICCV 2019** (IEEE/CVF) | ✅ VERIFIED | (novo, incluído na triagem 2026-05-25 como dataset complementar ao FairFace) |
| S9 | NISTIR 8280 | **Patrick J. Grother, Mei L. Ngan, Kayee K. Hanaoka** | Face Recognition Vendor Test (FRVT) Part 3: Demographic Effects | **NIST Interagency Report 8280**, dez/2019 (relatório técnico oficial) | ✅ VERIFIED | (novo, incluído na triagem 2026-05-25 como referência industry-wide; tipo: technical report, não peer-reviewed mas escala 200 algoritmos / 18M imagens) |

**Score Seção 1:** 9 verificados | 5 estavam ERRADOS | 4 estavam CORRETOS (incluindo S8 e S9 novos)

## Seção 2 — Papers de arquitetura e métodos

| # | arXiv | Autoria | Título exato | Venue | Status | Observações |
|---|---|---|---|---|---|---|
| A1 | arXiv:1512.03385 | **Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun** | Deep Residual Learning for Image Recognition | CVPR 2016 (arXiv tech report 10 Dec 2015) | ✅ VERIFIED | (ResNet — citação prévia correta) |
| A2 | arXiv:2201.03545 | **Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie** | A ConvNet for the 2020s | CVPR 2022 (arXiv v1 10 Jan 2022) | ✅ VERIFIED | (ConvNeXt — citação prévia correta) |
| A3 | arXiv:2004.11362 | **Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, Dilip Krishnan** | Supervised Contrastive Learning | NeurIPS 2020 (arXiv v1 23 Apr 2020) | ✅ VERIFIED | (SupCon — citação prévia correta) |
| A4 | arXiv:1801.07698 | **Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, Stefanos Zafeiriou** | ArcFace: Additive Angular Margin Loss for Deep Face Recognition | CVPR 2019 (versão TPAMI 2021 publicada) | ✅ VERIFIED | (citação prévia correta) |
| A5 | arXiv:2204.00964 | **Minchul Kim, Anil K. Jain, Xiaoming Liu** | AdaFace: Quality Adaptive Margin for Face Recognition | CVPR 2022 (Oral) | ✅ VERIFIED | (citação prévia correta) |
| A6 | arXiv:2103.06627 | **Qiang Meng, Shichao Zhao, Zhida Huang, Feng Zhou** | MagFace: A Universal Representation for Face Recognition and Quality Assessment | CVPR 2021 (Oral) | ✅ VERIFIED | (citação prévia correta) |
| A7 | arXiv:2203.16209 | **Sungho Park, Jewook Lee, Pilhyeon Lee, Sunhee Hwang, Dohyung Kim, Hyeran Byun** | **Fair Contrastive Learning for Facial Attribute Classification** | CVPR 2022 | ⚠️ CONTENT-TO-VERIFY | Anteriormente referenciado como "FSCL = Fair Supervised Contrastive Learning". O **título real é "Fair Contrastive Learning for Facial Attribute Classification"**. O acrônimo "FSCL" pode ser usado pelos autores no texto do paper para o método proposto — verificar na leitura integral antes de citar a sigla. |
| A8 | arXiv:2304.07193 | **Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo et al.** (lista longa) | DINOv2: Learning Robust Visual Features without Supervision | arXiv 2023 (versão estendida publicada em TMLR) | ✅ VERIFIED | (DINOv2 — citação prévia correta) |

## Seção 3 — Papers de fundamentação metodológica

### 3.1 Ensemble e reprodutibilidade

| # | arXiv / DOI | Autoria | Título exato | Venue | Status |
|---|---|---|---|---|---|
| F1 | arXiv:1612.01474 | **Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell** | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles | NIPS 2017 | ✅ VERIFIED |
| F2 | DOI:10.1109/34.58871 | **Lars Kai Hansen, Peter Salamon** | Neural Network Ensembles | IEEE Trans. on Pattern Analysis and Machine Intelligence (TPAMI), vol 12, no 10, pp. 993-1001, Oct 1990 | ✅ VERIFIED (sem arXiv — paper de 1990) |
| F3 | — | **Stuart Geman, Élie Bienenstock, René Doursat** | Neural Networks and the Bias/Variance Dilemma | Neural Computation, vol 4, no 1, pp. 1-58, 1992 | ⚠️ CONTENT-TO-VERIFY (sem arXiv; citação canônica disponível em DBLP/Google Scholar — paper de 1992) |
| F4 | IEEE Xplore 8995403 | **Deepak Bhaskaruni, Hui Hu, Chao Lan** | Improving Prediction Fairness via Model Ensemble | 2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI), pp. 1810-1814 | ✅ VERIFIED |
| F5 | arXiv:2003.12206 | **Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Larivière, Alina Beygelzimer, Florence d'Alché-Buc, Emily Fox, Hugo Larochelle** | Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program) | JMLR 2021 (arXiv v1 27 Mar 2020) | ✅ VERIFIED |
| F6 | arXiv:1709.06560 | **Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger** | Deep Reinforcement Learning that Matters | AAAI 2018 (arXiv v1 19 Sep 2017) | ✅ VERIFIED |
| F7 | arXiv:2103.03098 | **Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, Naz Sepah, Edward Raff, Kanika Madan, Vikram Voleti, Samira Ebrahimi Kahou, Vincent Michalski, Dmitriy Serdyuk, Tal Arbel, Chris Pal, Gaël Varoquaux, Pascal Vincent** | Accounting for Variance in Machine Learning Benchmarks | MLSys 2021 (submitted Mar 2021) | ✅ VERIFIED |

### 3.2 Equidade e mitigação

| # | arXiv / DOI | Autoria | Título exato | Venue | Status |
|---|---|---|---|---|---|
| F8 | arXiv:1706.04599 | **Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger** | On Calibration of Modern Neural Networks | ICML 2017 | ✅ VERIFIED |
| F9 | arXiv:1911.08731 | **Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang** | Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization | ICLR 2020 (arXiv v1 20 Nov 2019) | ✅ VERIFIED (Group DRO) |
| F10 | PMLR v81 | **Joy Buolamwini, Timnit Gebru** | Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification | Proceedings of the 1st Conference on Fairness, Accountability and Transparency (FAccT), PMLR 81:77-91, 2018 | ✅ VERIFIED |
| F11 | arXiv:1908.09635 | **Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, Aram Galstyan** | A Survey on Bias and Fairness in Machine Learning | ACM Computing Surveys 54.6 (2021) — arXiv v1 23 Aug 2019 | ✅ VERIFIED |
| F12 | arXiv:1807.00787 | **Till Speicher, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian Weller, Muhammad Bilal Zafar** | A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual & Group Unfairness via Inequality Indices | KDD 2018 | ✅ VERIFIED |
| F13 | arXiv:2011.02395 | **Tiago de Freitas Pereira, Sébastien Marcel** | Fairness in Biometrics: A Figure of Merit to Assess Biometric Verification Systems | IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM), 2022 | ⚠️ CONTENT-TO-VERIFY (autoria OK; descrição anterior associava o IR ao "produto FMR × FNMR" — o paper na verdade propõe **FDR (Fairness Discrepancy Rate)**. Reler antes de citar conteúdo) |

## Seção 4 — Convenções de citação para os materiais ativos

Padrão a seguir nos capítulos da dissertação e apresentações:

1. **Primeira menção em texto corrido**: nomes completos dos primeiros autores conforme verificado acima. Exemplo: *"O trabalho de AlDahoul, Tan, Kasireddy e Zaki (2024) reporta..."*
2. **Citações subsequentes**: *"AlDahoul et al. (2024)"*.
3. **Citações em parênteses**: *"(AlDahoul et al., 2024)"* ou *"(Pineau et al., 2021)"*.
4. **Em tabelas / notas técnicas**: forma curta *"AlDahoul 2024"* ou *"Pineau JMLR 2021"* é aceitável.
5. **Em BibTeX / referências finais**: usar a lista completa de autores conforme verificado.

## Seção 5 — Documentos contaminados (correção parcial aplicada)

Em 2026-05-25, antes da reorganização para `historico/`, foi feita
substituição textual global das 5 autorias incorretas em todos os 13
arquivos `.md` afetados (172 substituições no total). Assim, mesmo o
conteúdo arquivado em `historico/` não propaga mais as autorias
incorretas originais.

**Limitação reconhecida:** apenas autorias foram saneadas. Conteúdo
sintetizado a partir de abstracts pode estar parcialmente incorreto
em descrições de método, resultados e posicionamento. Materiais em
`docs/historico/` devem ser tratados como **referência histórica**, não
como fonte para novos textos. Qualquer reuso requer re-verificação.

## Seção 6 — Pendências e refinamentos por fazer

- [ ] **A7 (Park et al. CVPR 2022)**: verificar na leitura integral se
  os autores chamam o método de "FSCL" ou outro acrônimo. O título do
  paper é "Fair Contrastive Learning for Facial Attribute Classification".
- [ ] **F3 (Geman et al. 1992)**: paper canônico de bias/variance.
  Citação aceita amplamente; verificar via DBLP se citação mais
  precisa for necessária.
- [ ] **F13 (Pereira & Marcel 2022)**: verificar conteúdo na leitura
  integral. A descrição anterior do projeto associava o paper ao IR
  como "produto FMR × FNMR"; o paper na verdade propõe FDR (Fairness
  Discrepancy Rate). Caso a métrica original seja relevante para
  contextualizar nossa razão de disparidade, atualizar a discussão em
  documentos posteriores.
- [ ] Verificar outras citações tertiary (Breiman, Dietterich, Krizhevsky,
  Simonyan & Zisserman, Wolpert, Schapire, Crenshaw) **somente se** forem
  efetivamente citadas em algum material ativo. Por ora, mantidas como
  conhecimento canônico não-disputado.

## Seção 7 — Triagem editorial (Pesquisa Bibliográfica)

A Pesquisa Bibliográfica formal (ver `04_pesquisa_bibliografica/README.md`)
introduziu **dois critérios adicionais** de seleção de papers, exigidos
pelo orientador na reunião de 2026-05-25:

- **Critério 4 — Relevância editorial:** venue revisado por pares (lista
  detalhada em `04_pesquisa_bibliografica/README.md` §3.2). Preprints
  arXiv aceitos apenas com cobertura única ou impacto demonstrado.
- **Critério 5 — Impacto:** contagem de citações via Semantic Scholar +
  Google Scholar, modulada por idade do paper.

Os dados de triagem (venue confirmado + contagem de citações + decisão
de inclusão) dos papers desta Seção 1 + papers adicionais estão
catalogados em `04_pesquisa_bibliografica/_triagem.md`. Resumo da
Rodada 1 (2026-05-25):

| Paper desta seção | Citações S2 (2026-05-25) | Decisão |
|---|---|---|
| S1 (AlDahoul et al. 2024) | 11 | ✅ aprovado por exceção (cobertura única) |
| S2 (FairFace 2021) | ≥263 GS (S2 pendente por rate limit) | ✅ aprovado (dataset central) |
| S3 (FineFACE / Manzoor & Rattani 2024) | 2 | ✅ aprovado por exceção (proximidade temática) |
| S4 (U-FaTE / Dehdashtian et al. 2024) | 23 | ✅ aprovado |
| S6 (DSAP / Dominguez-Catena et al. 2024) | 7 | ✅ aprovado (venue top: Information Fusion) |
| S7 (Lafargue et al. 2025) | 1 | ✅ aprovado por exceção (cobertura única: EU AI Act) |
| S8 (RFW / Wang et al. 2019) | pendente | ✅ aprovado (venue top + dataset seminal) |
| S9 (NISTIR 8280 / Grother et al. 2019) | n/a (não-paper) | ✅ aprovado por exceção (technical report autoritativo) |

Toda atualização futura (mudança de venue, citações novas, papers
adicionais) é registrada em `_triagem.md` e refletida aqui após
verificação. `_triagem.md` é log; este documento é a fonte de verdade.

## Seção 8 — Reconhecimento de falha (registro)

Entre 2026-05-10 e 2026-05-24, foram produzidos ~20 documentos
contendo citações com autoria incorreta. A causa raiz foi **síntese de
conteúdo a partir de fetches superficiais sobre páginas HTML/abstract
do arXiv, sem verificação prévia de autoria**. A consequência foi que
o usuário apresentou ao orientador material contendo erros que ele
teve dificuldade de defender, gerando perda de credibilidade.

A política definida em **Seção 4** acima e na política documental
descrita em `docs/README.md` substitui essa prática a partir de
2026-05-25. Este documento é a **única fonte de verdade** para
citações nos materiais da dissertação.
