---
name: pereira-2026
status_verificacao: OVERVIEW_ONLY
autores: [Vitor Pereira Matias, Márcus Vinícius Lobo Costa, João Batista Neto, Tiago Novello de Brito]
ano: 2026
titulo: "Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild"
venue: "arXiv preprint (submetido março/2026)"
tipo_publicacao: preprint
arxiv_id: "2603.02475"
doi: null
url_primario: https://arxiv.org/abs/2603.02475
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-09
n_paginas: 12
lente_disrupcao: insumo
fonte_leitura: Apenas abstract via arXiv (PDF integral NÃO foi baixado/lido). Confirmação de meta-dados via WebFetch sobre arxiv.org/abs/2603.02475 em 2026-06-09.
---

> ⚠️ **AVISO METODOLÓGICO — ESTADO OVERVIEW_ONLY**
>
> Esta ficha foi construída **apenas a partir do abstract público no
> arXiv**, sem leitura do PDF integral. Não viola critérios de aprovação
> em [_triagem.md](_triagem.md#rodada-6) (R6-2), mas viola o padrão de
> rigor das demais fichas do corpus, que exigem leitura integral.
>
> **Cada seção marca explicitamente** se o conteúdo vem (a) do abstract
> verbatim/inferência direta, (b) de inferência razoável não verificada,
> ou (c) é pendência que requer PDF.
>
> **Para promover esta ficha a `VERIFIED`:**
> 1. Baixar PDF de https://arxiv.org/abs/2603.02475 para `pdfs/pereira_2026_skintonenet.pdf`
> 2. Ler integralmente
> 3. Reescrever as seções com conteúdo baseado em PDF, não em inferência
> 4. Atualizar `status_verificacao: VERIFIED` e `fonte_leitura`

# SkinToneNet + STW Dataset (Pereira Matias, Costa, Neto & Novello de Brito, 2026)

> Insumo direto da Etapa 1 do pipeline v3.2 — substitui o trabalho de
> treinar classificador MST do zero. **NÃO audita FairFace** (audita
> CelebA e VGGFace2), deixando a matriz MST × race 7-class do FairFace
> como contribuição original da nossa dissertação (C2).

## 1. Resumo do problema atacado

> *Fonte: abstract verbatim.*

A literatura de fairness em IA facial enfrenta limitações operacionais
para avaliar tom de pele:

- A escala Fitzpatrick 6-tons "lacks visual representativeness".
- Datasets MST públicos disponíveis são pequenos ou privados.
- Pipelines clássicos de CV existem mas poucos usam deep learning.
- Problemas conhecidos como train-test leakage e dataset imbalance.

Os autores propõem framework "comprehensive" para skin tone fairness
endereçando essas limitações simultaneamente.

## 2. Método

> *Fonte: abstract verbatim — três contribuições anunciadas. Detalhes
> internos do método (arquitetura específica, hiperparâmetros, splits,
> protocolo de anotação) NÃO estão no abstract.*

### 2.1 Contribuição 1 — Dataset STW (Skin Tone in the Wild)

- **42.313 imagens** de **3.564 indivíduos**.
- Anotação na escala **Monk Skin Tone 10-tons**.
- **Large-scale, open-access** (declarado pelos autores).

> **[PENDENTE PDF]** Protocolo de anotação (anotador único / consenso
> multi-anotador / IAA / Cohen's kappa). Splits treino/val/teste.
> Origem das imagens (datasets-fonte agregados). Licença específica.

### 2.2 Contribuição 2 — Benchmark Classic CV vs Deep Learning

- Linhas comparadas: **SkinToneCCV** (classic CV) versus deep learning.
- Achado declarado no abstract: *"classic models provide near-random
  results, while deep learning reaches nearly annotator accuracy."*

> **[PENDENTE PDF]** Quais arquiteturas exatas de deep learning estão
> no benchmark (CNNs específicos? múltiplos transformers?). Números
> de accuracy concretos. Definição operacional de "annotator accuracy"
> (média entre anotadores? maioria?).

### 2.3 Contribuição 3 — SkinToneNet

- **Backbone: ViT** (Vision Transformer) **fine-tuned**.
- Posicionado como **SOTA em generalização out-of-domain**.

> **[INFERÊNCIA RAZOÁVEL]** Pré-treino provavelmente em ImageNet
> (padrão para ViTs fine-tuned). Tamanho do ViT (Base/Small/Tiny)
> não declarado no abstract.
>
> **[PENDENTE PDF]** Tamanho exato do ViT. Detalhes de fine-tuning
> (LR, batch size, épocas, data augmentation). Ablations contra
> outros backbones (ConvNeXt, EfficientNet).

## 3. Datasets e setup experimental

> *Fonte: abstract.*

- **Treino**: STW (introduzido pelos autores).
- **Auditoria out-of-domain**: **CelebA** e **VGGFace2**.
- **FairFace NÃO mencionado** no abstract.

> **[PENDENTE PDF]** Splits exatos do STW. Tamanho dos conjuntos de
> auditoria (subset de CelebA/VGGFace2 usado). Protocolo de avaliação
> out-of-domain.

## 4. Métricas reportadas

> **[PENDENTE PDF]** O abstract não detalha métricas exatas além de
> referência genérica a "accuracy" e "annotator accuracy" como teto.
> Provável: accuracy de classificação 10-classe, possivelmente
> top-k accuracy ou tolerância a ±1 classe MST (comum em literatura
> de skin tone). NÃO verificado.

## 5. Resultados principais

> *Fonte: abstract verbatim.*

- **SkinToneNet atinge SOTA em skin-tone classification cross-domain**.
- **Modelos clássicos** (SkinToneCCV) **performam near-random**.
- **Deep learning atinge near-annotator accuracy**.
- **Auditoria de CelebA e VGGFace2** habilitada pelo classificador.

> **[PENDENTE PDF]** Números concretos. Tabelas de comparação.
> Distribuições MST exatas reportadas para CelebA e VGGFace2. Ablations.

## 6. Limitações declaradas pelos autores

> **[PENDENTE PDF]** Abstract não inclui seção de limitações.

## 7. Limitações que identifiquei (a partir do abstract apenas)

- **NÃO audita FairFace** — gap operacional para nossa dissertação,
  mas é também a abertura da nossa C2 como contribuição original.
- **Single arXiv preprint** — sem peer-review ainda (data: março/2026).
- **Citações no Semantic Scholar**: a verificar (paper muito recente).

> **[CAUTELA EPISTÊMICA]** A versão anterior desta ficha incluía mais
> limitações ("anotação sem consenso", "STW concentra sujeitos web",
> "apenas ViT-Small avaliado"). Todas eram INFERÊNCIAS NÃO VERIFICADAS
> a partir do abstract. Foram removidas para não criar falsa impressão
> de leitura profunda. Devem ser reavaliadas após leitura do PDF.

## 8. Relação com nossa pesquisa

> *Esta seção é a única em que mantemos análise extensa, porque o
> impacto na nossa tese é decidível a partir do abstract + verificação
> de meta-dados:*

### 8.1 Centralidade para o pipeline v3.2

Pereira et al. 2026 é central para a Rodada 6 porque pode substituir
uma das etapas do nosso pipeline:

| Antes da R6 | Possível após R6 (com Pereira) |
|---|---|
| Etapa 1: treinar classificador MST do zero | Etapa 1: usar **SkinToneNet pré-treinado** como insumo |

Decisão técnica registrada em [[../_validacao_cientifica_pipeline]]
seção 9.2: **usar SkinToneNet pré-treinado**, **condicional à
verificação** de:

1. Disponibilidade pública dos pesos do modelo (não declarada no abstract).
2. Licença permitindo uso acadêmico downstream.
3. Generalização adequada para FairFace (não auditado pelos autores).

> Se qualquer condição falhar, plano B: treinar classificador próprio
> com mesma arquitetura (ViT fine-tuned em STW se acessível, ou em
> Casual Conversations + MST-E como fallback).

### 8.2 Por que nossa C2 segue original

| Pereira 2026 audita | Nossa dissertação audita |
|---|---|
| CelebA, VGGFace2 — face attribute datasets gerais | **FairFace 7-class race** — dataset central de race classification fairness |
| Distribuições MST agregadas (inferido do abstract) | **Matriz cruzada MST × race** |

Resultado: SkinToneNet é **insumo**, não competidor da nossa C2 — desde
que nossa verificação confirme o status declarado.

### 8.3 Riscos de dependência externa

- **Disponibilidade dos pesos**: ainda não confirmada.
- **Versionamento**: registrar commit SHA + checksum dos pesos para
  reprodutibilidade.
- **Licença STW**: verificar antes de citar como "open-access" em
  publicação derivada.
- **Out-of-domain para FairFace**: o paper reporta generalização para
  CelebA e VGGFace2. FairFace tem perfil de coleta distinto. **Risco**:
  SkinToneNet pode performar abaixo do reportado em FairFace.
- **Mitigação**: validar SkinToneNet em subset FairFace anotado
  manualmente antes de confiar nos rótulos para análise quantitativa.

## 9. Pontos para citar

> Estes são rascunhos baseados no abstract. Devem ser revisados após
> leitura do PDF integral antes de aparecerem em publicação derivada.

- *"O dataset STW (Pereira et al., 2026 — preprint arXiv:2603.02475) —
  42.313 imagens de 3.564 indivíduos anotadas em Monk Skin Tone
  10-classe — propõe-se como base pública para treinar classificadores
  MST com generalização cross-domain. Esta dissertação investiga sua
  aplicabilidade como insumo da Etapa 1 do pipeline, reservando como
  contribuição original a auditoria sistemática da matriz MST × race
  classes sobre o FairFace, não coberta pelos autores."*

> **[PENDENTE PDF]** Outros pontos citáveis (números específicos de
> accuracy, comparações concretas) serão adicionados após leitura.

## 10. Arquivos relacionados

- PDF: **pendente** (`pdfs/pereira_2026_skintonenet.pdf` quando baixado).
- Código original: a verificar disponibilidade.
- Dataset STW: open-access declarado (URL exata pendente).

### Entradas relacionadas no corpus

- [[schumann_2023]] — MST scale + protocolo de consenso.
- [[dataset_hazirbas_2021]] — Casual Conversations (fallback se
  SkinToneNet não puder ser usado).
- [[dataset_karkkainen_2021]] — FairFace (gap que Pereira não cobre).
- [[buolamwini_2018]] — motivação histórica do uso de escalas de tom.
- [[dominguez_2024]] — DSAP (auditoria alternativa).
- [[perez_2018]] — FiLM (consome saída do SkinToneNet).
- [[../_validacao_cientifica_pipeline]] — análise da Rodada 6.

## 11. Trabalhos sugeridos pelos autores (Future Work)

> **[PENDENTE PDF]** Não verificável a partir do abstract.

## 12. Análise crítica do método

> **[BLOQUEADO]** Análise crítica honesta requer leitura do PDF
> integral. Esta seção será preenchida após:
>
> - Verificação do protocolo de anotação (consenso? IAA?)
> - Inspeção dos splits treino/val/teste
> - Verificação de hiperparâmetros e ablations
> - Comparação contra alternativas declaradas
> - Inspeção da inferência out-of-domain para CelebA e VGGFace2
>
> Análise crítica baseada em abstract seria especulação inadequada.
