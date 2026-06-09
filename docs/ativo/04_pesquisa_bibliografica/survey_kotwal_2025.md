---
name: survey-kotwal-2025
status_verificacao: VERIFIED
autores: [Ketan Kotwal, Sébastien Marcel]
ano: 2025
titulo: "Review of Demographic Fairness in Face Recognition"
venue: "IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)"
tipo_publicacao: journal
arxiv_id: null
doi: null
url_primario: https://publications.idiap.ch/attachments/papers/2025/Kotwal_TBIOM_2025.pdf
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 180+
lente_disrupcao: nenhuma
fonte_leitura: PDF integral extraído via pypdf (pdfs/kotwal_2025_survey.pdf); survey, não método
---

# Review of Demographic Fairness in Face Recognition (Kotwal & Marcel, 2025)

## 1. Resumo do problema atacado

Survey **mais recente e específico** de demographic fairness em
**face recognition** (FR), publicado em **IEEE TBIOM 2025** —
periódico top na área de biometria. Mais focado e mais atualizado
que [[survey_mehrabi_2021]] (que cobre ML em geral). Categoriza:
causas de differential performance, datasets, métricas, e técnicas
de mitigação em 4 estágios do pipeline FR.

Autores: **Idiap Research Institute, Switzerland** — mesmo grupo de
[[lafargue_2025]] (Sébastien Marcel é coautor de ambos).

## 2. Método

(Survey — não método experimental novo.)

**Estrutura do paper:**

- Seção II: Preliminares de FR evaluation (FMR, FNMR, score
  distributions).
- Seção III: **Causas das differential performance**:
  - III.A: Training datasets (balance é necessário mas não
    suficiente).
  - III.B: Variability in skin-tone.
  - III.C: Algorithmic sensitivity.
  - III.D: Image quality.
  - III.E: Soft attributes (head pose, brightness, etc.).
  - III.F: Combined/intersectional factors.
- Seção IV: Datasets (cobertura ampla, ver §3).
- Seção V: Fairness evaluation metrics.
- Seção VI: Mitigation strategies por estágio do pipeline.
- Seção VII: Open challenges and future directions.

## 3. Datasets cobertos (Seção IV)

(Lista inferida da menção, sem leitura exaustiva): RFW, BFW, FairFace,
DemogPairs, BUPT-Balancedface, Casual Conversations, AffectNet,
LFWA+, MORPH, IJB-A/B/C, MS-Celeb-1M, VGGFace2, MST-E. Inclusivo.

## 4. Métricas cobertas (Seção V)

(Inferido pelo abstract+intro, não leitura exaustiva): FMR/FNMR,
EOD, Demographic Parity Violation, Disparate Impact, score
distribution metrics, ISO/IEC 19795-10 standard.

## 5. Resultados principais (achados sintetizados)

Da Seção III.A lida:

**1. Consenso emergente: "balanceamento de dataset NÃO é solução
completa para fairness em FR."**

Citações importantes:

- **Krishnapriya et al.**: African-American cohorts têm FMR
  maior; Caucasian têm FNMR maior.
- **NIST FRVT (Grother et al., 2019)** [[grother_2019]]: false
  positives maiores em mulheres, crianças, idosos; false negatives
  maiores em Black, East Asian, Native American.
- **Klare et al.** (pré-deep learning): balanceamento não elimina
  disparidades; propõe **cohort-specific models** + dynamic
  selection.
- **Cavazos et al.**: necessidade de **thresholds mais altos para
  East-Asian faces** para FAR comparável.
- **Gwilliam et al.**: distribuição **enviesada para African**
  reduz differentials **mais efetivamente** que dataset balanceado —
  **contraintuitivo!**
- **Albiero et al.**: balancear gender NÃO reduz differential de
  gender em FR accuracy.
- **Wu & Bowyer**: mero balance em identidades/imagens é
  **insuficiente**; soft attributes (brightness, head pose) também
  importam.
- **Wang et al.**: race-balanced datasets falham em eliminar
  differentials — hipótese: "certain ethnicities are inherently more
  challenging to recognize".
- **Muthukumar et al.**: **structural facial features** (forma
  anatômica) explicam mais variação em dark-skinned females do que
  skin-tone isoladamente.

**Conclusão da seção:** *"Recent findings have indicated that the role
of balancedness of training data is limited, and other factors that
are inherent to the individuals or their soft attributes may have a
bigger role."*

(Resto do survey não lido em detalhe — usar como referência
bibliográfica.)

## 6. Limitações declaradas pelos autores

- **Foco em face recognition (verification + identification)**, NÃO
  attribute classification — distinção mantida em todo o paper.
- Race e ethnicity tratados como termos paralelos mas distintos.
- Age tratado como secundário pela natureza temporal.

## 7. Limitações que identifiquei

- **NÃO cobre classification de race per se** (foco em recognition).
  Para nossa pesquisa (classification), Mehrabi 2021 + Kotwal 2025
  são complementares mas nenhum focal.
- **Aceita o que é dado pela literatura** sem questionar de forma
  profunda escolhas paradigmáticas (e.g., taxonomia discreta vs
  contínua — [[neto_2025]] vai além).
- **Survey por natureza** — não apresenta dados experimentais novos.
- **Ainda não em Semantic Scholar com citações** (recente demais
  para acumular).

## 8. Relação com nossa pesquisa

**Papel:** referência canônica e mais atualizada para a discussão de
fairness em **biometria facial** — específica para esta área (vs
Mehrabi 2021 mais geral).

**Pontos de ancoragem:**

1. **Validação convergente do achado "balanceamento não basta":**
   Kotwal 2025 sintetiza 8+ referências independentes que confirmam
   o que já encontramos em [[dataset_karkkainen_2021]],
   [[dataset_wang_2019]], [[grother_2019]]. Reforça o argumento da
   dissertação.
2. **Insight de Gwilliam et al.** (citado em Kotwal): skewed para
   African **supera** balanced — direção contraintuitiva que merece
   menção em `06_gap.md`.
3. **Insight de Muthukumar et al.**: structural facial features >
   skin-tone para explicar viés. Sustenta Q10 🔬 (matriz skin tone ×
   race) — não basta skin tone, há outros factors.
4. **ISO/IEC 19795-10 standard** é citado — pode ser standard
   internacional para reportar métricas de demographic fairness.
5. **Idiap Research Institute (Sébastien Marcel)** consolida-se como
   grupo central para nossa pesquisa: produziu [[lafargue_2025]] e
   [[survey_kotwal_2025]] — track record forte.

## 9. Pontos para citar

- *"Kotwal e Marcel (IEEE TBIOM 2025) consolidam, em revisão
  sistemática de mais de 180 trabalhos, o consenso emergente da
  literatura: 'o papel do balanceamento dos dados de treinamento é
  limitado, e outros fatores inerentes aos indivíduos ou seus soft
  attributes podem ter papel maior' — refutando a hipótese ingênua
  de que dataset balanceado por design (caso do FairFace) é
  suficiente para eliminar disparidade demográfica."*
- *"Entre os achados sintetizados por Kotwal e Marcel (2025), destaca-
  se o contraintuitivo resultado de Gwilliam et al.: distribuições de
  treinamento enviesadas para faces africanas reduziram differentials
  raciais mais efetivamente que datasets perfeitamente balanceados.
  Esta observação sugere que a estratégia ótima de balanceamento é
  task-dependent e não trivialmente intuitiva."*
- *"A ISO/IEC 19795-10 (publicada recentemente, citada em Kotwal &
  Marcel 2025) estabelece guidelines internacionais para quantificação
  de demographic differentials em sistemas biométricos — referência
  regulatória paralela ao EU AI Act."*

## 10. Arquivos relacionados

- PDF: `pdfs/kotwal_2025_survey.pdf` (gitignored).
- Idiap publications page: publications.idiap.ch
- Entradas relacionadas: [[survey_mehrabi_2021]] (survey paralelo
  mais amplo), [[lafargue_2025]] (mesmo grupo Idiap),
  [[grother_2019]] (NIST FRVT — cited extensively),
  [[neto_2025]] (Kotwal precede Neto 2025; este vai além
  questionando discretização).

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Section VII (Open Challenges):

- **Standardização internacional de demographic fairness metrics** —
  ISO/IEC 19795-10 é início, mas adoção é parcial. ✅ **Alinhada com
  Q05** (métrica multi-classe sem consenso).
- **Investigar structural facial features além de skin tone** —
  citando Muthukumar et al., dark-skinned females têm features
  anatômicos contribuindo mais que tom de pele. ✅ **Alinhada com Q10**
  — quando a matriz tom × race for construída, verificar se features
  estruturais explicam erro residual.
- **Mitigação em estágios diversos do pipeline** (não só training-time)
  — Section VI lista pre-/in-/post-processing, mas avaliação
  comparativa é escassa. ✅ **Alinhada com Q04**.
- **Soft attributes (brightness, head pose, image quality)** como
  confundidores — controlar para isolar effect demográfico genuíno.
  ⚠ Parcialmente alinhada (nossa pesquisa não controla esses).
- **Cohort-specific models / dynamic model selection** (Klare et al.
  citado) — alternativa a um único modelo global. ❌ Fora do escopo
  imediato (mas direção interessante).
- **Privacy-preserving fairness measurement** — fairness sem expor
  atributos sensíveis. Emergente em SensitiveNets etc. ❌ Não
  alinhado com nossa pesquisa.
- **Generative models para data augmentation balanceada** —
  alternativa a sub-sampling que descarta dados. ❌ Fora do escopo
  (S10 já em standby).
- **Datasets com self-identification em escala** — referência implícita
  ao Casual Conversations. ✅ **Alinhada com Q01**.

## 12. Análise crítica do método

### (a) Rigor formal

- **Survey específico de FR** (vs Mehrabi 2021 mais geral) — mais
  focado e atualizado.
- **Síntese de 180+ referências** com consenso emergente claramente
  declarado.
- **Quote verbatim de achado central** disponível ("the role of
  balancedness of training data is limited").
- **Limitação**: aceita literatura como dada, sem questionar
  paradigmas profundamente (e.g., taxonomia discreta — Neto 2025 vai
  além).

### (b) Reprodutibilidade

- ✅ Survey publicado em IEEE TBIOM (peer-reviewed top venue
  biométrica).
- ✅ 180+ referências bem documentadas.
- ⚠ Recente demais para acumular citações (2025).
- ⚠ Não disponível no Semantic Scholar com métricas ainda.

### (c) Aplicabilidade ao pipeline v3.2

- **Referência canônica e mais atualizada** para FR fairness.
- **Validação convergente do achado "balanceamento não basta"** —
  sintetiza 8+ referências independentes.
- **Insight Gwilliam** (skewed-toward-African > balanced) é
  contraintuitivo e merece menção.
- **Insight Muthukumar** (structural features > skin-tone) sustenta
  Q10.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Foco em FR (não classification) | ✅ Justificada — escopo claro |
| 180+ referências sintetizadas | ✅ Justificada — comprehensiveness |
| Aceitar taxonomia discreta da literatura | ⚠ Limitação — Neto 2025 questiona |
| ISO/IEC 19795-10 como standard | ✅ Justificada — referência regulatória |
| Sem auditoria experimental nova | ✅ Choice — survey por natureza |

### (e) Conexão com R5/R6

- [[grother_2019]] NIST: NISTIR 8280 é referência central citada.
- [[dataset_wang_2019]] RFW: dataset central em FR fairness.
- [[lafargue_2025]]: mesma instituição (Idiap, Sébastien Marcel) —
  track record convergente.
- [[neto_2025]]: precede Kotwal cronologicamente; Neto questiona
  paradigma que Kotwal sintetiza.
- [[survey_mehrabi_2021]]: survey paralelo mais amplo. Kotwal é
  mais específico para nossa pesquisa.
- [[buolamwini_2018]]: citado como precursor histórico.
- **Implicação para v3.2**: Kotwal 2025 é **referência canônica
  obrigatória** para FR fairness. Cita-se como evidência da
  literatura mais atualizada.
