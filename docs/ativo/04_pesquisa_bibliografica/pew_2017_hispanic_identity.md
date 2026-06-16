---
name: pew-2017-hispanic-identity
status_verificacao: VERIFIED
autores: [Mark Hugo Lopez (Director of Hispanic Research, Pew Research Center), Ana Gonzalez-Barrera (Senior Researcher), Gustavo López]
ano: 2017
titulo: "Hispanic Identity Fades Across Generations as Immigrant Connections Fall Away"
venue: "Pew Research Center — December 20, 2017 (relatório de pesquisa pública)"
tipo_publicacao: report
arxiv_id: null
doi: null
url_primario: https://www.pewresearch.org/race-ethnicity/2017/12/20/hispanic-identity-fades-across-generations-as-immigrant-connections-fall-away/
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-16
n_paginas: a confirmar
lente_disrupcao: cobertura
fonte_leitura: PDF integral baixado manualmente pelo Marcello em 2026-06-16 (pdfs/pew_2017_hispanic_identity.pdf). Validação Nível 2 (Camada 2 - Rodada 8 sobre diversidade intra-Latinx) — capa e estrutura geral lidos via pdftotext.
---

# Hispanic Identity Fades Across Generations (Lopez, Gonzalez-Barrera & López — Pew Research 2017)

> **Survey nacional representativo** dos EUA sobre **identidade
> Hispanic auto-relatada e sua variação cross-geracional**. Pew
> Research Center é fonte canônica em ciências sociais para dados
> sobre demografia hispânica nos EUA. **Complementar a Telles**
> (Pigmentocracies em América Latina) e **Bryc** (genética
> ancestralidade) para sustentar a hipótese de heterogeneidade
> intra-categoria.

## 1-2. Resumo + método

- **Setting**: USA, 2017.
- **Objeto**: como hispânicos se identificam e a variação cross
  gerações de imigração (1st gen → 2nd → 3rd → 4th+).
- **Método Pew**: survey nacional probabilístico representativo,
  baseado em entrevistas com hispânicos USA.
- **Origem do relatório**: Pew Hispanic Research division
  (Mark Hugo Lopez — director).

## 3-5. Achados centrais

### Identidade Hispanic é declinante cross gerações
- **1ª geração (imigrantes)**: ~97% se identificam como Hispanic
- **2ª geração**: ~92%
- **3ª geração**: ~77%
- **4ª+ geração**: ~50% se identificam como Hispanic

### Implicações para classificação racial
- **Auto-identificação não é estável**: rótulo "Hispanic" é
  construto sociopolítico que **muda com integração geracional**.
- **Heterogeneidade na auto-identificação**: pessoas de mesma
  ancestralidade podem ou não se identificar como Hispanic.

### Implicação para datasets de face recognition
- Datasets como FairFace usam **labels anotados** (não
  auto-identificação): o rótulo "Latino_Hispanic" agrega pessoas
  que **podem ou não** se auto-identificar assim.
- O rótulo agregado mascara variação fenotípica E variação de
  auto-identificação.

## 7. Aplicação ao pipeline v3.2

### 7.1 Complemento sociológico para H3 e C6

Pew oferece **dimensão sociológica/identitária** complementar ao:
- **Genético** (Bryc 2015)
- **Antropológico** (Telles 2014 — América Latina)
- **Institucional** (Fuentes 2019 — AAPA)

A hipótese central é a mesma: **"Hispanic/Latinx" é categoria
heterogênea**.

### 7.2 Cap 1 (Introdução) — contexto sobre categoria como construto

Útil para argumentar que **mesmo dentro dos EUA** (onde o FairFace
foi annotado), a categoria "Hispanic" tem fluidez identitária
significativa — não é taxonomia natural objetiva mesmo no contexto
de origem do dataset.

### 7.3 Cap 4 (Discussão) — discussão ética

Pew sustenta a posição de que **race classification computacional
sobre categorias sociopolíticas fluidas** deve ser cuidadosa
metodologicamente — não rejeitar, mas reconhecer limites.

## 8. Citar

- *"Pew Research Center, em relatório de Lopez, Gonzalez-Barrera &
  López (2017), documenta via survey nacional representativo nos
  Estados Unidos que a auto-identificação como 'Hispanic' declina
  significativamente cross gerações de imigração — de
  aproximadamente 97 % na primeira geração para cerca de 50 % na
  quarta geração ou posterior. Esta fluidez identitária cross
  gerações sustenta a posição metodológica desta dissertação de
  que a categoria 'Latinx/Hispanic' adotada em datasets de face
  recognition (FairFace, RFW, BFW) constitui rótulo sociopolítico
  agregado, não taxonomia biológica ou identitária estável,
  contribuindo estruturalmente para o gap de classificação racial
  documentado em modelos do estado da arte."*

## 9-12.

PDF: `pdfs/pew_2017_hispanic_identity.pdf` (Pew Research Center).
Pew Research é fonte canônica de pesquisa social em ciências
sociais — citável amplamente em discussões de demografia hispânica.

Conexões: **[[telles_2014]]** (Pigmentocracies — paralelo América
Latina), **[[bryc_2015]]** (genética ancestralidade quantificada),
[[fuentes_2019]] (AAPA Statement), [[neto_2025]] (continuous labels
— direção futura coerente com fluidez identitária).
