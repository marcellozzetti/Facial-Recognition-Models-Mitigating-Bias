---
name: dataset-hazirbas-2021
status_verificacao: VERIFIED
autores: [Caner Hazirbas, Joanna Bitton, Brian Dolhansky, Jacqueline Pan, Albert Gordo, Cristian Canton Ferrer]
ano: 2021
titulo: "Towards Measuring Fairness in AI: the Casual Conversations Dataset"
venue: "IEEE/CVF CVPR Workshops (CVPRW) — TBIOM extended"
tipo_publicacao: conference_workshop
arxiv_id: "2104.02821"
doi: null
url_primario: https://arxiv.org/abs/2104.02821
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~50
lente_disrupcao: paradigma
fonte_leitura: PDF integral extraído via pypdf (pdfs/hazirbas_2021_casual.pdf)
---

# Casual Conversations Dataset (Hazirbas, Bitton, Dolhansky, Pan, Gordo & Canton Ferrer, 2021)

## 1. Resumo do problema atacado

Datasets faciais existentes (FairFace, UTKFace, LFWA+, CelebA) têm
problema metodológico fundamental: **atributos demográficos são
anotados por terceiros**, não pelo próprio sujeito. Para gênero e
idade especificamente, isto introduz viés do anotador. Hazirbas et
al. (Facebook AI / Meta) propõem **Casual Conversations**:
**self-reported gender and age** + **Fitzpatrick skin tone** anotado
por anotadores treinados + **rotulação de ambiente de iluminação**.
Aplicado para auditar bias dos top-5 vencedores do DeepFake Detection
Challenge (DFDC).

## 2. Método

### 2.1 Construção do dataset

- **3 011 sujeitos pagos** ("paid actors") com **consentimento
  explícito** para uso de imagem em AI.
- **>45 000 vídeos**, média de **15 vídeos por sujeito**.
- Múltiplos estados dos EUA.
- **v2 em 2023** (Porgali et al. — não detalhada aqui): 5 567
  participantes de 7 países.

### 2.2 Anotações

| Atributo | Método | Justificativa |
|---|---|---|
| **Age** | **Self-reported** | atributo objetivo; eliminar viés do anotador |
| **Gender** | **Self-reported** | identidade, não percepção (limita "Other" 0.1%) |
| **Fitzpatrick skin tone (I–VI)** | Anotadores treinados, voto majoritário sobre frames amostrados | atributo subjetivo, mas com escala dermatológica padrão |
| **Ambient lighting** | Marcado dark/light | controle de confounder |

### 2.3 Argumento explícito contra ethnicity labeling

Citação direta do paper (P 3):
> *"Public benchmarks tend to also provide annotated ethnicity labels.
> However, we find that labeling the ethnicity of subjects could lead
> to inaccuracies. Raters may have unconscious biases towards certain
> ethnic groups that may reduce the labelling accuracy."*

> *"In the FairFace [Kärkkäinen & Joo, 2021] dataset paper, the
> authors claim that skin tone is a one dimensional concept in
> comparison to ethnicity because lighting is a big factor when
> deciding on the skin tone... Although these claims sound reasonable,
> the ethnicity attribute is still subjective and can conceptually
> cause confusions in many aspects; for example, there may be no
> difference in facial appearance of African-American and African
> people, although, they may be referred to with two distinct racial
> categories. We, therefore, have opted to annotate the apparent skin
> tone of each subject."*

**Esta é uma defesa explícita de skin tone > race** na anotação.

## 3. Datasets e setup experimental

- **Casual Conversations**: o dataset proposto (3 011 sujeitos / 45K
  vídeos).
- **DFDC test set**: para auditar top-5 winners do DeepFake Detection
  Challenge.
- **Adience, IJB-A**: comparações de balanceamento (não usados em
  experimento direto).

## 4. Métricas reportadas

- **Accuracy / FNR / FPR por subgrupo demográfico** (age, gender,
  Fitzpatrick, lighting).
- DFDC: log-loss + precision/recall por subgrupo.

## 5. Resultados principais

**Achados-bandeira (sem números exatos na primeira leitura, mas
narrative consistente):**

1. **Top-5 DFDC winners apresentam degradação em darker skin tones**
   — confirma padrão Buolamwini & Gebru.
2. **Performance pior em low ambient lighting** — interage com skin
   tone (worst-case: darker skin + low light).
3. **Apparent age e gender classifiers** (Adience-trained) também têm
   gaps.

## 6. Limitações declaradas pelos autores

- **Gender binário** (M/F + Other 0.1%) — reconhece exclusão.
- Atores pagos → não é dataset in-the-wild.
- Fitzpatrick em vez de MST (MST não existia em 2021).
- Cobertura geográfica limitada a EUA na v1.

## 7. Limitações que identifiquei

- **NÃO inclui race labels.** Por design — defendem skin tone +
  self-reported gender/age como melhor abordagem.
- **Não diretamente comparável a FairFace** (taxonomias diferentes).
  Logo, não pode ser usado para validar/comparar nosso treino sobre
  FairFace race 7-class.
- **Self-reported age tem limitação:** validade da auto-percepção em
  faixa etária (alguns sujeitos podem reportar age "psicológica" vs
  "biológica").
- **Apparent skin tone via anotadores** ainda sujeito a inter-rater
  bias (precedente para Schumann 2023 que estuda exatamente isso).

## 8. Relação com nossa pesquisa

**Centralidade conceitual e metodológica:**

1. **Self-reported gold standard:** estabelece que anotação por
   terceiros (caso FairFace) tem **erro estrutural** em comparação
   com self-report. Justifica nossa **frente Q01 🔬** (auditar
   reliability das anotações FairFace para Latinx).
2. **Argumento contra ethnicity labeling** é direto e citável.
   Sustenta a discussão em `06_gap.md` sobre **limitações
   estruturais da taxonomia racial**.
3. **Multi-axis fairness** (age × gender × skin tone × lighting):
   precedente para análise multi-dimensional além de race apenas.
4. **Pivot natural para nossa Q10 🔬** (matriz skin tone × race):
   método de anotação Casual Conversations (anotadores treinados, voto
   majoritário sobre frames múltiplos) pode ser **importado** para
   anotar Fitzpatrick em subset de FairFace.

## 9. Pontos para citar

- *"Hazirbas, Bitton, Dolhansky, Pan, Gordo e Canton Ferrer (CVPRW
  2021) argumentam, na introdução do dataset Casual Conversations, que
  a anotação de etnia por terceiros 'pode levar a imprecisões — os
  avaliadores podem ter vieses inconscientes em relação a certos
  grupos étnicos que reduzem a precisão da rotulação'. Em vez disso,
  optam por (a) self-report para age e gender, e (b) Fitzpatrick para
  apparent skin tone — abordagem 'human-centered' que evita o viés
  de observador inerente à categorização racial."*
- *"O dataset Casual Conversations (3 011 sujeitos, 45 000 vídeos)
  estabelece um padrão metodológico que **nenhum dataset facial com
  rotulação racial atualmente atende**: combinação de auto-
  identificação (gender, age) com escala fenotípica anotada
  consistentemente (Fitzpatrick), além de controle explícito de
  iluminação ambiente como confundidor."*

## 10. Arquivos relacionados

- PDF: `pdfs/hazirbas_2021_casual.pdf` (gitignored).
- Repo Meta: github.com/facebookresearch/CasualConversations
- Entradas relacionadas: [[buolamwini_2018]] (precursor conceitual),
  [[schumann_2023]] (continua a investigação de skin tone annotation),
  [[dataset_karkkainen_2021]] (FairFace — alvo da crítica metodológica).
- **Responde parcialmente Q01 🔬, fundamenta Q09 e Q10 🔬** em
  [[_perguntas]].

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Section 5 (Conclusion) + Discussion sections:

- **Casual Conversations v2** (já publicada por Porgali et al. 2023):
  expandir para 7 países além dos EUA. Cobrir diversidade geográfica.
  ⚠ Parcialmente alinhada (não modifica nossa metodologia core).
- **Aplicar Casual Conversations a outras tarefas além de DFDC** —
  age/gender/emotion classification, face recognition. ✅ **Alinhada
  com Q10** — usar pipeline de anotação para FairFace seria adaptação
  direta.
- **Avaliar AI generative models** com mesma metodologia (text-to-image,
  AI avatars). ❌ Fora do escopo (geração).
- **Color-correction protocols** para improved inter-rater agreement
  em Fitzpatrick. ✅ **Alinhada com Q10** (anotação MST/Fitzpatrick
  manual exige protocolo similar).
- **Métodos de bias mitigation** usando dataset com self-reported
  attributes como ground truth. ✅ **Alinhada com Q01** — auditoria
  de anotações third-party (caso FairFace) contra self-reported é
  exatamente o gold-standard que falta.
- **Expandir para gender labels além do binário** — autores
  reconhecem limitação. ❌ Fora do nosso foco em race.
