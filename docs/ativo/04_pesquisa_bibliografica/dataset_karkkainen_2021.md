---
name: dataset-karkkainen-2021
status_verificacao: VERIFIED
autores: [Kimmo Kärkkäinen, Jungseock Joo]
ano: 2021
titulo: "FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation"
venue: "IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)"
tipo_publicacao: conference
arxiv_id: "1908.04913"
doi: null
url_primario: https://arxiv.org/abs/1908.04913
citacoes_google_scholar: 263+
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 76
lente_disrupcao: cobertura
fonte_leitura: PDF integral extraído via pypdf (pdfs/karkkainen_2021_fairface.pdf)
---

# FairFace (Kärkkäinen & Joo, 2021)

## 1. Resumo do problema atacado

Datasets faciais públicos pré-2019 são fortemente enviesados para faces
caucasianas (~80%); outras raças, especialmente Latino e Middle Eastern,
são sub-representadas. Esse desbalanceamento (a) gera modelos com
acurácia inconsistente entre grupos demográficos, (b) limita a
aplicabilidade de sistemas de análise facial a populações não-brancas e
(c) afeta adversamente pesquisas em ciências sociais que dependem de
inferência demográfica via imagens. O paper propõe um novo dataset
balanceado em raça/gênero/idade como **mitigação de viés via dados**,
não via algoritmo.

## 2. Método

### 2.1 Construção do dataset

- **Fonte primária:** YFCC-100M Flickr dataset (licenças Creative
  Commons Attribution + Share Alike); fontes secundárias: Twitter,
  newspapers online.
- **Estratégia de balanceamento:** coleta incremental, ajustada por
  composição demográfica estimada por país após uma rodada inicial de
  7 125 anotações. EUA e países europeus foram **excluídos** nas fases
  finais após saturação de faces brancas.
- **Detecção:** dlib (CNN-based). Tamanho mínimo de face: 50×50 pixels.
- **Anotação:** Amazon Mechanical Turk, **3 anotadores por imagem**;
  rótulo aceito se ≥2 concordam; se discordância total, re-anotação
  com 3 novos workers; se ainda sem acordo, descarte.
- **Refinamento:** treinamento de modelo sobre as anotações iniciais e
  reaplicação sobre o dataset; **re-verificação manual** apenas onde
  predição do modelo discordou da anotação humana. (Esta é uma forma
  de bootstrapping de qualidade — não documentam métrica de
  agreement final.)

### 2.2 Taxonomia de raça (7 categorias)

| Categoria | Notas |
|---|---|
| White | (origem: Census Bureau dos EUA) |
| Black | |
| Indian | subgrupo separado de Asian |
| East Asian | |
| Southeast Asian | distinção explícita de East Asian — primeira dataset large-scale a fazer |
| Middle East | primeira dataset large-scale a incluir como categoria separada |
| Latino | tratado como **raça** (não etnia), justificado por "judged from facial appearance" |

Categorias originalmente consideradas e **descartadas por amostragem
insuficiente**: Hawaiian/Pacific Islanders, Native Americans.

### 2.3 Setup de treinamento

- **Arquitetura:** ResNet-34 (única).
- **Otimizador:** ADAM, learning rate 0.0001.
- **Framework:** PyTorch.
- **Detecção de face em runtime:** dlib CNN-based.
- Não reportam batch size, número de épocas, augmentation, scheduler,
  weight decay, ou critério de seleção de checkpoint.

## 3. Datasets e setup experimental

### 3.1 Tamanho e split do dataset proposto

- **Total: 108 501 imagens** (face crops).
- Variantes do modelo treinado: Full, 18K, 9K (subsets para comparação
  justa com baselines menores).
- Splits train/val/test específicos não claramente declarados no texto
  — o paper foca no dataset como recurso; partição oficial fica no
  repositório GitHub (https://github.com/joojs/fairface).

### 3.2 Datasets comparados

- **UTKFace** (Zhang et al. 2017): 20K, com raça em 4 categorias
  (White, Black, Asian, Indian).
- **LFW A+** (Liu et al. 2015): 13K, derivado de LFW.
- **CelebA**: 200K, sem anotação de raça (usado só para gênero).
- **PPB** e **DiF**: tabela 1 menciona mas não usados em experimentos.

### 3.3 Datasets de generalização (out-of-domain)

Três conjuntos externos, anotados de novo via MTurk para raça/gênero/idade:

- **Geo-tagged Tweets**: 5 000 faces de França, Iraque, Filipinas, Venezuela.
- **Media Photographs**: 8 000 faces de tweets de @nytimes e outras 500 contas de mídia.
- **Protest Dataset**: 8 000 faces de imagens de protestos coletadas via Google.

## 4. Métricas reportadas

- **Accuracy** (por classe e overall).
- **Maximum accuracy disparity:** ϵ(Ŷ) = max_{j,k ∈ D} | log(P(Ŷ=Y|A=j) / P(Ŷ=Y|A=k)) |.
  Forma de log-ratio. Análogo da accuracy parity da literatura.
- **Standard deviation across races/ages** das acurácias.
- **Equalized odds** referenciado conceitualmente (eq. 1) mas não
  reportado numericamente como métrica de avaliação.

## 5. Resultados principais (valores numéricos)

### 5.1 Classificação de gênero — disparidade ε

| Modelo treinado em | Max accuracy disparity ε |
|---|---|
| **FairFace** | **0.055** |
| UTKFace | 0.127 |
| LFW A+ | 0.359 |
| CelebA | 0.166 |

Resultado-chave: modelo FairFace produz ~7× menos disparidade que LFW
A+ e ~3× menos que CelebA.

### 5.2 Classificação de raça (média sobre Twitter+Media+Protest, Tabela 6)

| Modelo | All | White | Non-White | Black | Asian | E Asian | SE Asian | Latino | Indian | Mid-East |
|---|---|---|---|---|---|---|---|---|---|---|
| **FairFace (Full)** | **0.815** | 0.928 | 0.639 | 0.815 | 0.883 | 0.764 | **0.376** | **0.247** | 0.611 | 0.742 |
| FairFace 18K | 0.800 | 0.917 | 0.588 | 0.779 | 0.856 | 0.685 | 0.355 | 0.279 | 0.502 | 0.625 |
| FairFace 9K | 0.774 | 0.885 | 0.564 | 0.756 | 0.827 | 0.641 | 0.315 | 0.281 | 0.531 | 0.544 |
| UTKFace | 0.674 | 0.815 | 0.479 | 0.702 | 0.507 | — | — | 0.555 | — | — |
| LFW A+ | 0.684 | 0.969 | 0.348 | 0.395 | 0.497 | — | — | — | — | — |

**Achado crítico (sub-representação propaga-se mesmo no dataset
balanceado):** o modelo FairFace **mantém disparidade enorme entre
White (.928) e Latino (.247)** — uma diferença de **68 p.p.** Isso
acontece **apesar** do balanceamento do dataset. Causa provável:
dificuldade intrínseca de definir Latino visualmente (heterogeneidade
genética) + provável ruído de anotação MTurk em categorias menos
familiares aos anotadores.

### 5.3 Cross-dataset (Tabela 3, White só / Tabela 4, non-White)

| Treinado em | Race Acc (FairFace test, White) | Race Acc (FairFace test, non-White†) |
|---|---|---|
| FairFace | 0.937 | 0.754 |
| UTKFace | 0.800 | 0.693 |
| LFW A+ | 0.879 | 0.541 |

† **Aviso crítico:** o rodapé da Tabela 4 declara explicitamente
*"FairFace defines 7 race categories but only 4 races (White, Black,
Asian, and Indian) were used in this result to make it comparable to
UTKFace."*

**Consequência:** os números cross-dataset publicados pelo próprio
paper **não testam classificação de raça em 7 categorias no-domain
contra outros datasets**. Os números 7-class só aparecem na Tabela 6
(generalização para Twitter/Media/Protest, anotados pelos próprios
autores). Isso significa que **a tarefa de raça 7-class no-domain
contra baselines comparáveis fica órfã na literatura**.

## 6. Limitações declaradas pelos autores

- *"It is infeasible to balance across all possible co-occurrences"*
  (citando Hendricks et al. 2018). Balanceamento perfeito é impossível
  in-the-wild.
- *"Race is not a discrete concept and needs to be clearly defined"* —
  reconhecem fragilidade conceitual da taxonomia.
- Skin color (ITA) é insuficiente como proxy de raça: variação
  intra-grupo grande, sensível à iluminação, unidimensional vs raça
  multidimensional. (Justificativa para anotação humana.)
- Hawaiian/Pacific Islanders e Native Americans foram descartados por
  amostragem insuficiente.
- t-SNE com embedding dlib (treinado em datasets enviesados) pode
  enviesar a visualização de diversidade.

## 7. Limitações que identifiquei (leitura crítica)

- **Ausência de partição train/val/test explícita no paper.** A
  partição está no repositório GitHub, mas reprodutibilidade ficaria
  mais forte com declaração textual de tamanhos por split.
- **Ausência de hiperparâmetros completos.** Não reportam batch size,
  épocas, weight decay, scheduler, augmentation. Reprodução exata é
  impossível sem inspecionar código do repositório.
- **Critério de seleção de checkpoint não declarado.** Validation
  loss? Validation accuracy? Best epoch? Resultados de Tabelas 5 e 6
  não trazem ± desvio padrão por seed, então não é possível separar
  ruído de treinamento de efeito real.
- **MTurk como ground truth para raça é problemático.** O paper menciona
  ≥2/3 agreement como critério, mas não reporta inter-annotator
  agreement (κ de Cohen ou equivalente) por categoria. Para Latino e
  Middle Eastern, suspeita-se que o agreement seja menor que para
  White/Black — o que explicaria parte da disparidade .247 vs .928.
- **Maximum accuracy disparity ε** é definida em log-ratio mas
  apresentada em valor absoluto sem intervalo de confiança. Para uma
  métrica fundamentada em paridade estatística, faltam testes de
  significância.
- **Cross-dataset evaluation em 4 raças, não 7.** O paper introduz
  7-class mas não testa cross-dataset em 7-class — exige
  re-anotação dos outros datasets, que os autores não fizeram. Esta é
  uma lacuna metodológica explícita.
- **Anotação adicional pelos próprios autores nos datasets de
  generalização (Twitter/Media/Protest)** introduz potencial viés de
  anotação concordante com a taxonomia FairFace. Falta validação
  cruzada por anotadores independentes.

## 8. Relação com nossa pesquisa

**Centralidade:** este é o **dataset central** da dissertação. Todo
nosso trabalho experimental treina e avalia sobre FairFace.

**Pontos de ancoragem:**

1. **Taxonomia 7-class adotada integralmente** (White, Black, Indian,
   East Asian, Southeast Asian, Middle East, Latino).
2. **Caveat metodológico crítico:** Tabela 4 do paper FairFace mostra
   que cross-dataset em 7-class **não é testado pelos autores**. Nossa
   pesquisa enfrenta o mesmo limite — opera **in-domain** (FairFace
   train → FairFace val/test), com 7 classes. Esta é uma característica
   alinhada com a literatura **disponível**, não uma escolha por
   conveniência.
3. **Disparidade Latino (.247) e SE Asian (.376) no próprio
   FairFace-Full** confirma que **balanceamento de dataset não
   resolve fairness**. Esta é uma motivação direta para abordagens
   algorítmicas (mitigação in-processing, post-processing, calibração).
   O gap entre White (.928) e Latino (.247) é o tipo de "60+ pp gap"
   que nossas técnicas precisam atacar.
4. **Ausência de hiperparâmetros declarados no paper** justifica nossa
   ênfase em protocolo casado (3 seeds, hiperparâmetros explícitos)
   como contribuição metodológica complementar.
5. **Métrica ε (max accuracy disparity, log-ratio)** é distinta da
   nossa **razão de disparidade (max/min)**. Worth registering a
   conversão: max(Acc)/min(Acc) ≈ exp(ε). Ou seja, nosso DR=1.06 do
   FairFace seria ε≈0.058 nesta notação — coerente com o ε=0.055
   reportado para gênero.

## 9. Pontos para citar / posicionar

- *"O FairFace, único dataset large-scale a separar Southeast Asian
  como categoria distinta e a incluir Middle Eastern (Kärkkäinen &
  Joo, 2021), constitui a base empírica desta dissertação."*
- *"O próprio paper-fonte do FairFace não publica resultados de
  classificação de raça em 7 categorias contra outros datasets
  (Kärkkäinen & Joo, 2021, Tabela 4), reportando comparações
  cross-dataset apenas após mesclagem das 7 categorias em 4. Esta
  lacuna é estrutural: outros datasets simplesmente não oferecem as
  mesmas 7 categorias."*
- *"A disparidade entre White (acc ≈ 0.928) e Latino (acc ≈ 0.247)
  observada no próprio modelo FairFace-Full sobre validação externa
  (Kärkkäinen & Joo, 2021, Tabela 6) demonstra empiricamente que
  balanceamento de dataset é **necessário mas não suficiente** para
  equidade entre grupos raciais."*

## 10. Arquivos relacionados

- PDF local: `pdfs/karkkainen_2021_fairface.pdf` (gitignored)
- Texto extraído (pypdf): `pdfs/karkkainen_2021_fairface.txt` (gitignored)
- Entradas relacionadas: [[buolamwini_2018]] (motivação fundadora),
  [[wang_2019]] (RFW, dataset complementar), [[aldahoul_2024]] (única
  referência publicada para nossa tarefa exata).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S2.

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Section 5 (Conclusion):

- **Treinar modelos diversos sobre FairFace** — autores entregam o
  dataset, sugerem que pesquisadores usem. ✅ **Alinhada com toda
  literatura subsequente**.
- **Verificar fairness em outros classificadores comerciais** —
  ResNet-34 é baseline; outros backbones devem ser testados.
  ✅ **Alinhada com Q06**.
- **Estender taxonomia para incluir Hawaiian/Pacific Islanders e
  Native Americans** — autores descartaram por amostragem
  insuficiente. ❌ Fora do nosso escopo (escala de dados).
- **Investigar métricas fairness alternativas** — autores reportam
  ε (log-ratio max disparity), outras formulações possíveis.
  ✅ **Alinhada com Q05**.
- **Combinar FairFace com técnicas de mitigação algorítmica** —
  paper deixa em aberto o que fazer com o dataset balanceado.
  ✅ **Alinhada com Q04** — gap CENTRAL nossa.
- **NÃO** mencionam continuous labels ou cross-reference com skin
  tone. ✅ **Q09 e Q10 são frentes genuinamente abertas** —
  autores do dataset não cobrem.

## 12. Análise crítica do método (revisão pós-reunião 2026-06-04)

### (a) Rigor formal

- **Métrica ε (max accuracy disparity log-ratio)** é matematicamente
  bem definida mas **não tem intervalo de confiança reportado**.
  Para n=1500 imagens por subgrupo, IC de 95% sobre accuracy seria
  aproximadamente ±1.5 pp — o ε reportado de 0.055 pode ter overlap
  estatístico com 0.078 (UTKFace), enfraquecendo a conclusão de
  superioridade.
- Conexão com [[hardt_2016]]: ε é log-ratio de TPR; pode ser visto
  como variante de EO_h. **Conversão para EO_h padrão facilitaria
  comparação com literatura subsequente** (Park 2022, Sagawa 2020).

### (b) Reprodutibilidade

- **Crítico:** hiperparâmetros não declarados no paper. Apenas
  "ADAM + lr=0.0001". Faltam: batch size, épocas, weight decay,
  scheduler, augmentation, image size pré-processamento,
  early-stopping criterion.
- Split train/val: **não declarado textualmente**, só via GitHub.
- Sem multi-seed — single run reportado.
- **Implicação para nossa Fase 5**: precisamos **decidir
  hiperparâmetros e documentá-los explicitamente** em
  `02_metodologia.md` para reprodução.

### (c) Aplicabilidade ao pipeline v3.2

- **Dataset é a base do nosso pipeline** — não há "aplicabilidade
  questionável" no sentido convencional. **Mas há caveat
  metodológico**: como o paper não declara hiperparâmetros, o
  baseline 72% (Lin 2022, AlDahoul 2024) usa **escolhas que reproduzem
  empiricamente** o original, não escolhas garantidas idênticas.
- Para v3.2 (MST + race), FairFace **não tem labels MST** — exige
  classifier auxiliar ([[schumann_2023]] API). Trade-off: API tem
  viés próprio.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| 7 categorias raciais (incluir Middle East + Latino) | ✅ Justificada (cobertura de subgrupos não representados em outros datasets) |
| Descartar Hawaiian/Pacific Islanders + Native American | ✅ Justificada (sample insuficiente) |
| MTurk como anotador (não especialista) | ❌ Assumida — sem comparação com expert annotation |
| 3 anotadores + voto majoritário | ⚠ Implícita — sem reportar inter-rater agreement (κ) por classe |
| Padding=0.25 vs 1.25 (versões diferentes) | ✅ Justificada (versões para diferentes propósitos) |

### (e) Conexão com R5

- [[hardt_2016]]: ε do FairFace é variante de EO_h log-ratio. Tradução
  para EO_h padrão é direta.
- [[zemel_2013]]: FairFace é **apenas dataset**, não método de
  representação fair. Sua adoção em Fair Representation Learning
  (Park 2022, Madras 2018) é pós-hoc.
- [[kleinberg_2017]]: a métrica ε do FairFace satisfaz "balance for
  positive class" mas não calibration. Confirma teorema da
  impossibilidade — escolha de ε é decisão de design.
