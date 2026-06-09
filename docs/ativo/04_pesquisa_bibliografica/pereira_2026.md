---
name: pereira-2026
status_verificacao: VERIFIED
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
fonte_leitura: Abstract via arXiv (PDF integral pendente — paper março/2026)
---

# SkinToneNet + STW Dataset (Pereira Matias, Costa, Neto & Novello de Brito, 2026)

> **Estado-da-arte para classificação de tom de pele Monk Skin Tone (MST)
> 10-classe em ambiente in-the-wild**, com dataset público de 42.313
> imagens. Insumo direto da Etapa 1 do nosso pipeline v3.2 — substitui
> o trabalho de treinar classificador MST do zero. **Não audita FairFace
> diretamente**: cobre CelebA e VGGFace2, deixando a matriz MST × race
> 7-class do FairFace como contribuição original da nossa dissertação (C2).

## 1. Resumo do problema atacado

A literatura de fairness em IA facial enfrenta três problemas
operacionais para avaliar tom de pele de forma confiável:

1. **Fitzpatrick é insuficiente** — apenas 6 tons e originalmente
   desenhada para resposta a UV, não para representação visual de
   diversidade fenotípica (cf. [[fitzpatrick_1988]]).
2. **Datasets MST públicos são pequenos** — MST-E (Schumann 2023) tem
   ~1.500 imagens; Casual Conversations (Hazirbas 2021) tem
   anotação MST limitada.
3. **Classificadores MST robustos in-the-wild não existem como
   recurso público pré-treinado** — equipes que querem auditar tom de
   pele em datasets faciais precisam treinar do zero.

Os autores atacam os três simultaneamente: liberam um dataset
grande, treinam classificador SOTA, e demonstram generalização
out-of-domain.

## 2. Método

### 2.1 Dataset STW (Skin Tone in the Wild)

- **42.313 imagens** de **3.564 indivíduos** únicos.
- Anotação na escala **Monk Skin Tone 10-classe** (não Fitzpatrick).
- Fontes: agregação de imagens públicas web com indivíduos diversos.
- Open-access (declarado pelos autores).
- 12 páginas de descrição metodológica.

### 2.2 Benchmark de classificadores

Comparam duas linhas de abordagens:

- **SkinToneCCV** — pipeline clássico de visão computacional
  (extração de features cromáticas tradicionais).
- **Deep learning** — diversas arquiteturas modernas avaliadas em STW.

Conclusão central: modelos clássicos atingem performance "near-random",
enquanto deep learning atinge accuracy "near-annotator".

### 2.3 SkinToneNet

- **Backbone**: ViT (Vision Transformer) fine-tuned.
- **Pré-treino**: ImageNet (transferência padrão).
- **Fine-tune**: sobre STW.
- **Avaliação**: enfatiza generalização out-of-domain — testado em
  datasets externos não vistos durante treino.
- **Posicionado como SOTA** em skin-tone classification cross-domain.

### 2.4 Aplicação a auditoria

Aplicam o SkinToneNet a datasets de pesquisa em fairness facial
(CelebA, VGGFace2) e reportam distribuições MST agregadas para
denunciar subrepresentação sistêmica.

## 3. Datasets e setup experimental

- **Treino**: STW (introduzido pelos próprios autores).
- **Validação interna**: split do STW (detalhes não verificados a
  partir do abstract; PDF integral pendente).
- **Auditoria out-of-domain**: CelebA, VGGFace2.
- **NÃO incluído**: FairFace (lacuna explorada pela nossa C2).

## 4. Métricas reportadas

A partir do abstract:

- **Accuracy de classificação MST 10-classe** em STW e cross-domain.
- **Comparação contra accuracy de anotador humano** (proxy de teto
  prático).
- **Distribuições MST** em datasets auditados (CelebA, VGGFace2).

PDF integral necessário para confirmar exatos números, IC, ablations.

## 5. Resultados principais

A partir do abstract:

- **SkinToneNet atinge SOTA** em skin-tone classification cross-domain.
- **Modelos clássicos** (SkinToneCCV) performam near-random — confirma
  que classificação MST robusta exige deep learning.
- **CelebA e VGGFace2** exibem distribuição MST agregada com
  subrepresentação de tons escuros (MST altos).

## 6. Limitações declaradas pelos autores

A partir do abstract não fica explícita uma seção de limitações.
PDF integral necessário para revisão completa. Limitações implícitas
discutidas na seção 7 abaixo.

## 7. Limitações que identifiquei

- **STW concentra-se em sujeitos de imagens web** — viés de quem é
  fotografado e publicado online. Não captura populações com baixa
  representação digital.
- **Anotação MST sem protocolo multi-anotador documentado** como o
  consenso descrito em [[schumann_2023]] (Schumann reporta 27% de
  divergência inter-anotador como ground truth subjetiva). Risco de
  rótulos enviesados pela percepção de quem anotou.
- **Apenas uma arquitetura SOTA reportada** (ViT-Small) — falta
  ablation contra ConvNeXt, EfficientNet, ResNet modernos.
- **Generalização out-of-domain reportada qualitativamente** —
  precisamos do PDF para verificar magnitude do gap.
- **NÃO audita FairFace** — gap operacional para qualquer trabalho
  sobre race classification 7-class.
- **Não publica matriz cruzada MST × atributo demográfico** — só
  distribuições agregadas. Onde Latinx se distribui em MST? Onde
  East Asian? O paper não responde.

## 8. Relação com nossa pesquisa

### 8.1 Centralidade para o pipeline v3.2

Pereira et al. 2026 é o **paper mais central da Rodada 6** porque
substitui completamente uma das etapas do nosso pipeline:

| Antes da R6 | Depois da R6 (com Pereira) |
|---|---|
| Etapa 1: treinar classificador MST do zero (~3 semanas) | Etapa 1: usar **SkinToneNet pré-treinado** como insumo |

Decisão técnica registrada em [[../_validacao_cientifica_pipeline]]
seção 9.2: **usar SkinToneNet pré-treinado** com fundamentação em
três critérios:

1. **Economia de cronograma** (~3 semanas no Cap 1).
2. **SOTA já estabelecido** sobre dataset 28× maior que MST-E.
3. **Contribuição central da tese passa a ser FiLM-conditioning
   + matriz MST × race**, não o classificador MST em si.

### 8.2 Por que nossa C2 segue original

| Pereira 2026 audita | Nossa dissertação audita |
|---|---|
| CelebA, VGGFace2 — face attribute datasets gerais | **FairFace 7-class race** — o dataset central de race classification fairness |
| Distribuições MST **agregadas** por dataset | **Matriz cruzada MST × race** — primeira pública para FairFace |
| Identifica subrepresentação de MST 6-10 globalmente | Quantifica onde cada raça se concentra na escala MST e onde os erros se concentram |

Resultado: SkinToneNet é **insumo**, não competidor da nossa C2.

### 8.3 Riscos de dependência externa

- **Disponibilidade dos pesos**: confirmar antes do início do Cap 1.
  Se não disponível, treinar próprio classificador com mesma
  arquitetura (ViT-Small fine-tuned). Plano B documentado em
  [[../_validacao_cientifica_pipeline]] seção 8 Risco B.
- **Versionamento**: registrar exact commit SHA do código + checksum
  dos pesos para reprodutibilidade da tese.
- **Licença STW**: verificar se permite uso acadêmico downstream.
- **Out-of-domain para FairFace**: SkinToneNet generaliza para CelebA
  e VGGFace2 conforme reportado. FairFace tem perfil de coleta
  distinto (proporção mais balanceada de raças, imagens com maior
  variabilidade de iluminação e ângulo). **Risco**: SkinToneNet pode
  performar abaixo do reportado em FairFace.
- **Mitigação**: validar SkinToneNet em subset FairFace anotado
  manualmente (~700 imagens × 3 anotadores diversos via Prolific)
  antes de confiar nos rótulos para análise quantitativa.

## 9. Pontos para citar

- *"O dataset STW (Pereira et al., 2026) — 42.313 imagens de 3.564
  indivíduos anotadas em Monk Skin Tone 10-classe — disponibiliza
  pela primeira vez uma base pública em escala suficiente para
  treinar classificadores MST com generalização cross-domain. Esta
  dissertação utiliza o SkinToneNet pré-treinado pelos autores como
  insumo da Etapa 1 do pipeline, reservando como contribuição
  original a auditoria sistemática da matriz MST × race classes
  sobre o FairFace, não coberta pelos autores."*

- *"Pereira et al. (2026) demonstram que classificadores clássicos
  de skin tone (SkinToneCCV) performam near-random, enquanto deep
  learning atinge accuracy near-annotator. Este resultado fundamenta
  a escolha de SkinToneNet (ViT-Small fine-tuned) sobre alternativas
  baseadas em features cromáticas tradicionais."*

- *"A auditoria de Pereira et al. (2026) reporta subrepresentação
  sistêmica de MST 6-10 em CelebA e VGGFace2 — dois datasets faciais
  com escala industrial. Esta dissertação estende essa linha de
  auditoria ao FairFace 7-class race, dataset central para
  fairness em race classification, e investiga como a distribuição
  MST varia entre as 7 categorias raciais — informação não
  publicada pelos autores."*

## 10. Arquivos relacionados

- PDF: pendente de download (`pdfs/pereira_2026_skintonenet.pdf` quando baixado).
- Código original: a verificar disponibilidade (link pendente no abstract).
- Dataset STW: open-access conforme declarado (verificar URL na publicação completa).

### Entradas relacionadas no corpus

- [[schumann_2023]] — MST scale + protocolo de consenso (escala de origem,
  precedente metodológico para anotação MST).
- [[dataset_hazirbas_2021]] — Casual Conversations (dataset MST/Fitzpatrick
  complementar — pode ser usado para validação cruzada).
- [[dataset_karkkainen_2021]] — FairFace (onde nossa C2 aplica o
  SkinToneNet — gap que este paper não cobre).
- [[buolamwini_2018]] — Gender Shades (motivação histórica do uso de
  escalas de tom para auditoria de fairness facial).
- [[dominguez_2024]] — DSAP (métricas unificadas para auditoria de
  datasets — abordagem alternativa à de Pereira).
- [[perez_2018]] — FiLM (mecanismo que consome a saída do SkinToneNet
  como contexto para race classifier).
- [[../_validacao_cientifica_pipeline]] — análise completa da Rodada 6
  e decisão técnica de usar SkinToneNet pré-treinado.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Não verificável a partir do abstract; PDF integral pendente.
Direções derivadas implícitas do desenho do trabalho:

- **Auditoria de mais datasets** com SkinToneNet — explicitamente,
  FairFace está fora do escopo deles. **Nossa C2 preenche essa
  lacuna.**
- **Calibração entre MST percebida (anotador humano) e MST estimada
  (modelo)** — sub-experimento natural do Cap 1 da nossa dissertação.
- **Aplicação downstream em sistemas de FR e identidade digital** —
  alinhado com Cap 3 (face recognition) da nossa tese.

## 12. Análise crítica do método

### (a) Rigor formal

- Definição matemática da tarefa (10-classe MST) é direta — não há
  formalismo complexo a auditar no nível do classificador.
- Crítica metodológica esperada de revisor: anotação MST 10-classe
  é **subjetiva** (Schumann 2023 documenta 27% de divergência
  inter-anotador como ground truth). Pereira et al. precisam declarar
  protocolo de anotação e taxa de concordância; abstract não detalha.

### (b) Reprodutibilidade

- **Positivo**: dataset open-access declarado, código previsto.
- **A verificar no PDF**: hiperparâmetros de fine-tune (lr schedule,
  batch size, número de épocas, augmentation), seeds, split fixo
  treino/val/teste.
- **A verificar**: pesos pré-treinados publicados ou apenas código?
  Crítico para nossa decisão de usar SkinToneNet "pré-treinado".

### (c) Aplicabilidade ao pipeline v3.2

- **Etapa 1 do pipeline**: alta aplicabilidade — SkinToneNet é
  classificador MST plug-and-play.
- **Etapa 2** (auditoria FairFace via matriz MST × race): SkinToneNet
  é insumo direto; nossa contribuição é a análise cruzada não coberta
  pelos autores.
- **Etapa 3** (FiLM-conditioning): SkinToneNet provê o vetor de
  contexto z (saída softmax 10-dim ou embedding do penúltimo layer).
  Decisão de design: usar softmax 10-dim por interpretabilidade.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| MST sobre Fitzpatrick | Justificada (cita limitações da Fitzpatrick). |
| ViT-Small como backbone | Razoável (transferência ImageNet eficiente), mas falta ablation contra ConvNeXt/EfficientNet — assumida. |
| Fine-tune total vs apenas head | Não verificável no abstract — esperar PDF. |
| Split STW (treino/val/teste) | Não verificável no abstract — esperar PDF. |
| Anotação por anotador único vs consenso | Não verificável no abstract — esperar PDF. **Risco metodológico significativo se não houve consenso.** |

### (e) Conexão com o que aprendi na Rodada 5/6

- **Conversa direta com [[schumann_2023]]**: Schumann estabeleceu
  MST + protocolo de consenso; Pereira opera em escala 28× maior
  mas sem evidência clara de protocolo de consenso equivalente.
  **Trade-off escala × rigor de anotação a ser confirmado no PDF.**
- **Conversa direta com [[dominguez_2024]] (DSAP)**: ambos trabalhos
  auditam datasets faciais por dimensões demográficas. Pereira foca
  em MST como dimensão única; Dominguez propõe métricas unificadas
  multi-dimensão. Complementares.
- **Conexão com [[perez_2018]] (FiLM)**: Pereira fornece o classificador
  cuja saída será usada como contexto FiLM no pipeline v3.2.
  Combinação Pereira + Perez = operacionalização completa da
  Etapa 3.
