---
name: lafargue-2025
status_verificacao: VERIFIED
autores: [Valentin Lafargue, Emmanuelle Claeys, Jean-Michel Loubes]
ano: 2025
titulo: "Fairness is in the Details: Face Dataset Auditing"
venue: "ECML PKDD 2025 (Springer LNCS vol 16022)"
tipo_publicacao: conference
arxiv_id: "2504.08396"
doi: null
url_primario: https://arxiv.org/abs/2504.08396
citacoes_google_scholar: null
citacoes_semantic_scholar: 1
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: 47
lente_disrupcao: metodologica
fonte_leitura: PDF integral extraído via pypdf (pdfs/lafargue_2025_fairdetails.pdf), 20+ páginas
---

# Fairness is in the Details (Lafargue, Claeys & Loubes, 2025)

## 1. Resumo do problema atacado

O **EU AI Act** (vigente em 2024) exige que sistemas de IA sob
auditoria comprovem **conformidade com princípios de fairness, equity
e transparência** via análise sistemática de viés. Para datasets de
imagem isso impõe um desafio prático: (a) variáveis sensíveis
(ethnicity, age, gender) não vêm rotuladas; (b) **anotação manual é
inviável** em escala; (c) usar CNN para anotar automaticamente
introduz **incerteza** que os testes estatísticos clássicos (χ², KS,
Wasserstein) ignoram. O paper propõe um **pipeline completo de
auditoria** em 2 fases:

1. Extração de variáveis sensíveis via CNN customizada (gender, age,
   Fitzpatrick skin type) com poucos labels manuais.
2. **Statistical test uncertainty-aware** que considera o erro do
   classifier ao testar disparidades.

## 2. Método

### 2.1 Variáveis sensíveis adotadas

- **Gender:** binário (M/F), por limitação dos datasets disponíveis
  (autores explicitamente reconhecem como limitação ética).
- **Age:** grupos.
- **Fitzpatrick skin type:** **5 classes** (I, II, III-IV merged, V,
  VI). III-IV fundidos por inter-annotator agreement baixo em não-
  especialistas.
- Justificativa para Fitzpatrick vs race: estabilidade conceitual
  (origem dermatológica) + popularidade pós-Gender Shades.
- **NÃO** adota Census 2000 (Black/White/Latino/Asian/Other),
  considerada restritiva.
- **NÃO** adota Monk Skin Tone Scale (10 classes), considerada
  difícil de separar em grupos para teste estatístico.

### 2.2 Extração de features (Fase 1)

- **Backbone:** ResNet-50 ou ResNet-101 + custom classification head.
- **Pré-processamento alternativas testadas:**
  1. Imagem original.
  2. Background removido (rembg).
  3. **Apenas pele segmentada** (DeepLabV3 + MobileNetV3 Large
     pretrained em Celeb-HQ).
- **ITA (Individual Typology Angle)** computado dos pixels de pele:
  ITA = arctan((L* − 50) / b*) × (180/π)
  onde L* = luminância, b* = componente yellow/blue (CIELAB).
- **Para gender e age:** usa o modelo pretreinado do FairFace
  (ResNet-34) — declarado como melhor performance disponível.
- **Para Fitzpatrick:** fine-tuning custom da ResNet com
  ITA + skin-related info concatenados na latent layer (15-dim feature
  vector adicional).

### 2.3 Teste estatístico (Fase 2)

**Duas formulações:**

- **Parity test (1 variável sensível):** H₀: X₀ ~ distribuição alvo
  (uniforme para gender; world distribution real para age e
  Fitzpatrick — não uniform!). Testa se a distribuição observada
  difere da alvo.
- **Equal representation test (2 variáveis sensíveis):**
  H₀: P(X₀ | Xⱼ = grupo) = P(X₀ | Xⱼ ≠ grupo). Testa se a distribuição
  condicional de X₀ depende de Xⱼ.

**3 testes implementados:**

- **χ²** (categórico).
- **CLT-based mean test** (para variáveis contínuas).
- **Wasserstein-based two-sample test**.

### 2.4 Uncertainty-aware: o componente novo

- Modelo CNN tem accuracy ≠ 100%; algumas predições estão erradas.
- Procedimento: **permutar aleatoriamente** algumas predições
  automáticas (na proporção do erro do classifier), mantendo
  anotações manuais. Repetir N vezes → distribuição de p-values.
- **Decisão:** rejeitar H₀ se a **mediana dos p-values < threshold**.
- Resultado: testa estatística robusta ao erro do extrator de
  features. **Crítico** porque testes ingênuos rejeitam H₀ por
  artefatos do classifier, não por viés real.

## 3. Datasets e setup experimental

- **Generated Photos dataset:** 10 000 imagens sintéticas geradas via
  GAN, com **rótulos Census 2000 balanceados** (~equal Caucasian,
  Asian, Hispanic/Latino, Black).
- **CelebA subset:** 1 500 imagens amostradas aleatoriamente.
- **Anotação manual:** 3 anotadores não-especialistas. Maioria
  decide. Para gender, "perceived gender" — limitação binária
  reconhecida.

## 4. Métricas reportadas

- **p-values** dos 3 testes (χ², mean, Wasserstein).
- **Mediana sobre N simulações** (uncertainty-aware).
- **CNN accuracy/precision** para classificação de Fitzpatrick (no
  Appendix).
- **Distribuição empírica vs alvo** plotada graficamente.

## 5. Resultados principais (valores numéricos)

(Não totalmente lidos no excerto; ficam no Section 6 do paper.)

**Achados conceituais:**

- ITA tem **correlação forte com Fitzpatrick** mas não é 1:1 —
  ITA é unidimensional, Fitzpatrick agrega múltiplos fenotípicos
  (textura de cabelo, formato de rosto, reação UV). Confirma
  argumento de Gender Shades.
- Treinar CNN customizado com **poucos labels manuais** + ITA como
  feature suplementar atinge acurácia comparável a abordagens com
  datasets muito maiores.
- **Pipeline open-source** em github.com/ValentinLafargue/FairnessDetails.

(Para números exatos, ler Section 6.)

## 6. Limitações declaradas pelos autores

- **Gender binário:** anotadores tiveram que escolher M/F; reconhece
  exclusão de transgender, bigender. Trabalho inicial.
- **Multi-modal categórico tratado como binary one-vs-all:** perde
  estrutura de classes correlacionadas. Limitação conhecida.
- **Disparate Impact (DI)** poderia ser aplicado mas autores optaram
  por não usar para evitar confusão com fairness do modelo CNN
  auditando.
- **Testes baseados em mediana de p-values sobre simulações** podem
  ser computacionalmente caros para datasets muito grandes.

## 7. Limitações que identifiquei (leitura crítica)

- **Datasets pequenos para auditoria:** 10K (Generated Photos) e 1.5K
  (CelebA subset). Auditar datasets de ML em escala industrial (>1M
  imagens) é o caso de uso real; aqui é prova-de-conceito.
- **Generated Photos é sintético.** A validação em dataset sintético
  controlado é útil mas **não substitui** validação em dataset real
  com viés conhecido (e.g., FairFace, RFW). Falta benchmark cruzado.
- **Fitzpatrick com 5 classes (III-IV mescladas)** é **mais coarse**
  que o original de 6 classes. Reduz poder discriminatório.
- **Uncertainty-aware** assume que o erro do classifier é
  **aleatório**. Mas em prática, erros são **sistemáticos**
  (e.g., classifier erra mais em Latino que em White — Gender
  Shades). Permutar aleatoriamente subestima o viés real do extrator.
- **Anotadores não-especialistas:** Fitzpatrick III-IV "rarely reach
  full agreement" entre não-experts. Mesclar resolve o problema
  estatístico mas perde resolução conceitual. Anotação por
  dermatologista (como em Gender Shades) seria mais rigoroso.
- **Sem race direta:** o paper opta por Fitzpatrick (justifica
  cuidadosamente), mas isso impede comparação direta com literatura
  que usa raça (FairFace, RFW). É uma **escolha conceitual válida**
  mas limita transposição.
- **Sem análise da própria CNN auditora:** o paper audita datasets
  mas não audita explicitamente o **viés do classifier que extrai
  features**. Possível recursividade não tratada.

## 8. Relação com nossa pesquisa

**Lente principal:** Lafargue et al. é um **paper de auditoria de
datasets**, não de mitigação algorítmica. Para a dissertação, a
relevância é em três frentes:

1. **EU AI Act como motivação institucional:** o paper é a única
   referência verificada que ancora discussão de fairness em
   **regulação concreta** (EU AI Act 2024). Útil para introdução /
   contexto ético da dissertação.
2. **Uncertainty-aware statistical test** é uma contribuição
   metodológica geral: **testes estatísticos clássicos rejeitam H₀
   por artefatos de measurement error, não viés real**. Esta lição
   se aplica à nossa pesquisa: o nosso DR (max F1 / min F1) também
   sofre de variance do classifier; reportar intervalos de confiança
   via bootstrap seria a importação direta dessa lição.
3. **ITA + skin segmentation as features:** abordagem técnica
   sofisticada para skin tone. Útil como referência se a dissertação
   eventualmente investigar mitigação ortogonal a raça (e.g.,
   fairness by Fitzpatrick em vez de race) — porém não está no
   escopo atual.

**Limitação de aplicabilidade direta:**

- Lafargue auda **datasets**, não **classifiers**. Nossa pesquisa
  trabalha sobre classifiers treinados no FairFace.
- Lafargue usa **Fitzpatrick**, não race. Nossa pesquisa usa race
  7-class.
- Citação será **conceitual/regulatória**, não experimental.

## 9. Pontos para citar / posicionar

- *"Com a entrada em vigor do European AI Act em 2024, sistemas de
  classificação facial passam a estar sujeitos a requisitos formais
  de auditoria de fairness, equity e transparência (Lafargue et al.,
  2025). Esta exigência regulatória reforça a urgência de
  metodologias rigorosas para quantificação de viés demográfico em
  sistemas faciais."*
- *"Lafargue, Claeys e Loubes (2025), publicado nos anais do ECML
  PKDD 2025, propõem um pipeline de auditoria de datasets que combina
  extração de features sensíveis (Fitzpatrick skin type) via CNN com
  testes estatísticos uncertainty-aware. A contribuição metodológica
  central é a observação de que testes estatísticos clássicos (χ²,
  Wasserstein) rejeitam H₀ por artefatos de erro do classifier, não
  por viés genuíno do dataset — uma lição transponível para qualquer
  cálculo de disparidade demográfica baseado em rótulos preditos
  automaticamente."*
- *"A escolha de Lafargue et al. (2025) pela escala Fitzpatrick em
  detrimento de categorias raciais (Census 2000) reforça a tensão
  conceitual entre estabilidade da medida (fenotípica) e
  aplicabilidade social (racial) já registrada por Buolamwini e Gebru
  (2018) — tensão essa que a presente dissertação opta por resolver
  na direção oposta, adotando a taxonomia racial em sete categorias
  do FairFace (Kärkkäinen & Joo, 2021), em consonância com as
  aplicações de ciências sociais que motivam o trabalho."*

## 10. Arquivos relacionados

- PDF local: `pdfs/lafargue_2025_fairdetails.pdf` (gitignored).
- Texto extraído: `pdfs/lafargue_2025_fairdetails.txt` (gitignored).
- Código: https://github.com/ValentinLafargue/FairnessDetails.
- Entradas relacionadas: [[buolamwini_2018]] (Gender Shades — fonte
  conceitual da escolha Fitzpatrick),
  [[dominguez_2024]] (DSAP — outra metodologia de auditoria de
  datasets, complementar),
  [[dataset_karkkainen_2021]] (FairFace usado como gender/age
  classifier no pipeline).
- Referência canônica: `docs/ativo/00_referencias.md` Seção 1, linha S7.
