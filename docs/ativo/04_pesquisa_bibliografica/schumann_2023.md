---
name: schumann-2023
status_verificacao: VERIFIED
autores: [Candice Schumann, Gbolahan O. Olanubi, Auriel Wright, Ellis Monk Jr., Courtney Heldreth, Susanna Ricco]
ano: 2023
titulo: "Consensus and Subjectivity of Skin Tone Annotation for ML Fairness"
venue: "NeurIPS 2023 Datasets and Benchmarks Track"
tipo_publicacao: conference
arxiv_id: "2305.09073"
doi: null
url_primario: https://arxiv.org/abs/2305.09073
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~86
lente_disrupcao: paradigma
fonte_leitura: PDF integral extraído via pypdf (pdfs/schumann_2023_mst.pdf)
---

# Consensus and Subjectivity of Skin Tone Annotation for ML Fairness (Schumann, Olanubi, Wright, Monk Jr., Heldreth & Ricco, 2023)

## 1. Resumo do problema atacado

Anotações de **tom de pele percebido** são essenciais para auditoria
de fairness mas **subjetivas** (afetadas por iluminação, contexto
cultural do anotador, etnia inferida do sujeito). O paper estabelece
o **Monk Skin Tone Scale (MST)** — escala de **10 pontos** desenvolvida
por Ellis Monk (Harvard) em parceria com Google — como alternativa
mais inclusiva à Fitzpatrick (6 pontos). Libera **MST-E dataset**
(1 515 imagens + 31 vídeos) para treinamento de anotadores.

**Achado central:** anotadores de **diferentes regiões geográficas**
têm **variação sistemática** em interpretação do MST — implica
necessidade de pool diverso + alta replicação.

## 2. Método

### 2.1 Monk Skin Tone Scale (MST)

- **10 pontos** (vs 6 do Fitzpatrick).
- Desenvolvida por **sociólogo Ellis Monk** (Harvard).
- Designed for **broader range of skin tones**.
- Critério explícito: Fitzpatrick é **enviesado para pele clara**
  (3 das 6 categorias cobrem "perceived as White").

### 2.2 MST-E Dataset

- **1 515 imagens** + **31 vídeos** de 19 sujeitos.
- Anotações expert (incl. Monk) + diversos pools de crowdworkers.
- Released como **training reference** para futuros anotadores.

### 2.3 Estudos de consenso

- **Pool de fotógrafos profissionais** vs **crowdworkers treinados**.
- **Regiões geográficas diferentes** (USA vs índia/etc.).
- Mede: inter-annotator agreement, deviation from expert ground truth,
  systematic regional differences.

## 3. Datasets e setup experimental

- **MST-E** como ground truth de treinamento.
- Avaliação em **diversas geografias** de anotadores.
- **NÃO testa em FairFace** — foco em estabelecer o protocolo MST.

## 4. Métricas reportadas

- **Agreement com expert** (Monk).
- **Cross-region variation** (Δ médio por par de regiões).
- **Replication count effect** (curva de convergência com N).

## 5. Resultados principais

1. **Crowdworkers treinados podem anotar MST com confiabilidade
   próxima a expert**, sob condições controladas.
2. **Anotadores de regiões diferentes têm "mental models" diferentes
   de MST** — variação sistemática (não ruído aleatório).
3. **Recomendação:** usar **pool diverso de anotadores** + **higher
   replication count** (≥3) por imagem para fairness research.
4. **MST > Fitzpatrick** para inclusividade — mas ambos exigem
   anotação cuidadosa.

## 6. Limitações declaradas pelos autores

- **Tradeoff granularidade × consenso:** scales muito finas (36-point
  Felix von Luschan) reduzem inter-rater agreement.
- **Self-report vs annotated** — paper não defende fortemente um
  sobre o outro (cita Casual Conversations como exemplo de
  self-report viable).
- **Não substitui ground truth dermatológico** (que exige avaliação
  física, não foto).

## 7. Limitações que identifiquei

- **MST-E é dataset pequeno** (1 515 imagens, 19 sujeitos) — útil
  para treinamento de anotadores, não para training de modelos
  de classification em escala.
- **Não testa em FairFace** — gap claro para nossa pesquisa.
- **Validade ecológica** (anotação consistente em condições
  variáveis de iluminação) discutida mas não exaustivamente
  testada.
- **Bias do anotador inferindo race a partir do skin tone** é
  reconhecido mas não quantificado.

## 8. Relação com nossa pesquisa

**Centralidade extrema para Q10 🔬 (matriz skin tone × race):**

1. **MST como padrão moderno:** se vamos anotar skin tone em FairFace,
   MST é a escala correta (não Fitzpatrick), conforme Google/Monk.
2. **Protocolo de anotação** está documentado: pool diverso (≥3
   regiões geográficas), N ≥ 3 anotadores por imagem, MST-E como
   training reference. **Diretamente importável** para nosso
   experimento Q10.
3. **Inter-annotator agreement por classe** — mensurar κ_classe é
   exatamente o que Q01 🔬 propõe para Latinx em FairFace. Schumann
   2023 dá o template metodológico.
4. **Defesa do skin tone como métrica de fairness** — alternativa a
   race que pode aparecer no `06_gap.md` e `07_thesis_statement.md`.

## 9. Pontos para citar

- *"Schumann et al. (NeurIPS Datasets and Benchmarks 2023) demonstram
  que anotações do Monk Skin Tone Scale por anotadores de diferentes
  regiões geográficas exibem variação sistemática — não meramente
  ruído aleatório — sugerindo que 'mental models' de tom de pele são
  culturalmente moldados. A recomendação operacional é usar pool
  diverso de anotadores e taxa de replicação maior que o padrão para
  pesquisa de fairness."*
- *"A escala MST (Monk Skin Tone, 10 pontos), proposta por Ellis Monk
  em parceria com Google (Schumann et al., 2023), supera a Fitzpatrick
  (6 pontos, originalmente desenvolvida para dermatologia, com viés
  para pele clara) em granularidade e inclusividade, sendo
  particularmente recomendada para tarefas de fairness em visão
  computacional."*

## 10. Arquivos relacionados

- PDF: `pdfs/schumann_2023_mst.pdf` (gitignored).
- MST-E dataset: skintone.google
- Entradas relacionadas: [[buolamwini_2018]] (Fitzpatrick precursor),
  [[dataset_hazirbas_2021]] (Casual Conversations + Fitzpatrick),
  [[lafargue_2025]] (cita MST mas adota Fitzpatrick).
- **Responde Q01 🔬 (template metodológico), fundamenta Q10 🔬**.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Conclusão e Discussion:

- **Estender MST-E para mais sujeitos e geografias** — 19 sujeitos em
  MST-E é insuficiente para training reference em escala global.
  ❌ Fora do escopo (criação de dataset).
- **Investigar interação com tarefas downstream** — paper só estuda
  anotação per se, não impacto em modelos de classification.
  ✅ **Alinhada com Q10** — exatamente isso (MST → impacto em race
  classification).
- **Comparar MST vs Fitzpatrick em scoring de fairness real** —
  declaração: "ambos exigem anotação cuidadosa" mas qual produz
  melhor sinal de bias? Aberto. ✅ **Alinhada com Q10**.
- **Modelos que incorporam perspectiva de anotador individual** (não
  só consenso) — cita literatura emergente (Goyal, Aroyo, Díaz).
  ⚠ Parcialmente alinhada — direção sociotécnica, complemento ao
  nosso work.
- **Reduzir cognitive load em scales muito granulares** (Felix von
  Luschan 36-pt) — sugestão para outros pesquisadores; MST como
  compromisso. ❌ Fora do escopo.
