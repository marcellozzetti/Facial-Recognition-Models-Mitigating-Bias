# Gabarito conceitual — domínio para a defesa da qualificação

**Data:** 31/08/2026
**Motivação:** feedback do orientador — todos os conceitos-chave da
metodologia proposta precisam estar 100 % sob meu domínio. Este
documento consolida as perguntas mais prováveis da banca com respostas
preparadas, ancoradas em referências.

**Escopo:** 4 blocos — (1) Vision Transformers e ConvNeXt-T,
(2) FiLM, (3) Metodologia científica (ablation, controle, hipóteses),
(4) Métricas de fairness.

---

## 1. Vision Transformers (ViT) e ConvNeXt-T

### 1.1 O que é um Vision Transformer? (Dosovitskiy et al. 2020)

**Resposta essencial:**
> "É a adaptação do mecanismo de self-attention dos Transformers (originalmente
> NLP) para imagens. A imagem é dividida em patches (tipicamente 16×16),
> cada patch é linearizado e embedado como um 'token', assim como palavras
> em NLP. Um token especial [CLS] é adicionado e sua representação final
> serve como agregado da imagem para classificação. Positional encodings
> preservam a informação espacial."

**Fluxo interno:**
1. Imagem 224×224 → 196 patches de 16×16
2. Cada patch → vetor 768-dim (linear embedding)
3. + token [CLS] + positional encodings
4. Sequência passa por N blocos de self-attention (12 no ViT-B/16)
5. Estado final do [CLS] → cabeça de classificação

**Vantagem:** captura dependências globais desde a primeira camada
(atenção all-to-all).
**Custo:** exige mais dados de treinamento e mais compute que CNNs.

### 1.2 Por que ConvNeXt-T e não ResNet ou ViT? (Liu et al. 2022)

**Resposta essencial:**
> "ConvNeXt-T é a modernização convolucional da linha ResNet: mantém
> a estrutura hierárquica em 4 estágios, mas adota práticas dos ViTs
> (depthwise 7×7, LayerNorm, ativação GELU, inverted bottleneck).
> Escolhi por 4 critérios objetivos:"

1. **Paridade com ViTs a custo convolucional** — 82 % top-1 em ImageNet, comparável ao Swin-T (81.3 %) com menor custo de inferência.
2. **Estabilidade de fine-tuning** — LayerNorm no lugar de BatchNorm é robusto a variações de batch size (crítico para 3 sementes × 3 configurações).
3. **Compatibilidade com FiLM** — os 4 estágios hierárquicos são pontos de inserção naturais para FiLM.
4. **Comparabilidade científica** — ResNet-34 é o baseline canônico do FairFace (Kärkkäinen 2021); ConvNeXt-T como controle arquitetural moderno isola o efeito do FiLM sobre a modernização do backbone.

### 1.3 Perguntas prováveis

**"Por que não ViT-B/16 diretamente?"**
> "ViT-B tem 86 M params vs 28 M do ConvNeXt-T. Para nosso protocolo (3 sementes × 3 configs × 200-400 h GPU total), ViT-B triplicaria o orçamento sem ganho documentado em fairness facial. Além disso, ViT não tem estrutura hierárquica clara — inserir FiLM entre 'estágios' exigiria adaptações arbitrárias. ConvNeXt-T oferece os 4 estágios naturais."

**"Por que não Swin Transformer?"**
> "Swin-T tem estrutura hierárquica e também poderia acomodar FiLM. Escolhemos ConvNeXt porque (a) é convolucional puro, casando com a linhagem histórica ResNet do FairFace, (b) demonstra que ganho não vem de mudança de paradigma para atenção — se atenção fosse necessária, seria confundido com o efeito do FiLM."

**"E o ResNet-34 do FairFace original?"**
> "Serve como baseline-âncora da literatura, não como método principal. Usar ResNet-34 puro impossibilita comparação com métodos modernos e limita a discussão. Nossa Config A (ConvNeXt-T puro) atua como controle arquitetural moderno, isolando o efeito do condicionamento."

---

## 2. FiLM — Feature-wise Linear Modulation (Perez et al. 2018)

### 2.1 O que é FiLM?

**Resposta essencial:**
> "FiLM é um mecanismo de condicionamento arquitetural que modula
> features intermediárias de uma rede neural por meio de uma
> transformação afim canal a canal:
>
> **F′ = γ ⊙ F + β**
>
> Onde F é a feature map, γ é um vetor de escala e β um vetor de
> deslocamento, ambos gerados por MLPs a partir do sinal condicionante
> z (no nosso caso, o vetor softmax MST 10-dim). É como se o sinal
> externo 'consultasse' a rede e dissesse: 'para essa entrada, amplifique
> esse canal, diminua aquele, e desloque a distribuição desses aqui'."

### 2.2 Por que FiLM e não outras técnicas de condicionamento?

Tabela comparativa (aderente ao §Alternativas de condicionamento do Cap. 2):

| Alternativa | Por que descartada |
|---|---|
| Concatenação direta MST → features | Explosão paramétrica proporcional à largura do feature map; dilui o sinal |
| Conditional Batch Normalization | Caso particular do FiLM (generalizado por Perez 2018) |
| Cross-attention | ~3× o custo paramétrico do FiLM para sinal 10-dim, sem ganho demonstrável |
| AdaIN (Adaptive Instance Normalization) | Projetado para style transfer, incompatível com sinal demográfico global |
| SPADE | Requer mapa espacial denso; vetor MST é global |
| HyperNetworks | Instabilidade documentada + sobredimensionado para nosso porte |
| LoRA / Adaptadores | Modificam pesos da rede, não condicionam features — categoria diferente |

### 2.3 Perguntas prováveis

**"Como você garante que FiLM não perturba o backbone pré-treinado no início?"**
> "Inicializo o gerador MLP de γ com pesos zerados e viés = 1, e o gerador de β com pesos zerados e viés = 0. Assim, no início do treino, γ=1 e β=0 para qualquer z, tornando a camada FiLM matematicamente equivalente à identidade. O backbone ImageNet fica intacto na primeira época e o FiLM aprende gradualmente. Isso é o 'init identidade' — implementado e testado (12 unit tests em test_film.py)."

**"Qual o overhead paramétrico?"**
> "~380 mil parâmetros, cerca de 1,3 % dos 28 M do ConvNeXt-T. Substancialmente inferior a alternativas como cross-attention (~3× o custo do FiLM). Testado empiricamente em test_wrap_convnext_film_overhead_below_2pct."

**"O sinal de entrada é softmax MST — por que não one-hot?"**
> "Softmax preserva incerteza: se o classificador MST tem 40 % de probabilidade para o tom 5, 35 % para tom 6, 25 % para tom 4, o FiLM recebe essa distribuição e modula de forma proporcional. One-hot descartaria essa informação, forçando decisão dura. Reforço externo: Balancing Beyond Discrete Categories (arXiv:2506.01532, 2025) confirma que tratar demográfico como contínuo é preferível a categorização discreta."

**"E se FiLM não ajudar? Como você sabe se o problema é do FiLM ou do sinal MST?"**
> "É exatamente para isolar essas hipóteses que temos as 3 configurações do ablation:
> - Config A → B: com/sem FiLM (mantendo tudo constante) — isola o efeito do MECANISMO.
> - Config B → C: FiLM+MST vs FiLM+CLIP-text — isola o efeito da NATUREZA DO SINAL.
> Se B > A, mas C > B, o sinal semântico (CLIP) é melhor que MST direto. Se B > A > C, MST é o sinal certo. Se A ≈ B, FiLM não ajuda com esse sinal."

### 2.4 Precedentes de FiLM que fortalecem nossa escolha

- **Perez et al. 2018 (AAAI)** — paper original, aplicação em VQA.
- **Dumoulin et al. 2018 (Distill)** — revisão canônica das variantes.
- **FADiff — Nature Sci Rep 2025** — usa FiLM em face recognition em longa distância (precedente RECENTE, próximo do nosso domínio).
- **Perez FiLM Generator** — hipernetwork para gerar (γ, β) — arquitetura padrão.

---

## 3. Metodologia científica

### 3.1 O que é ablation study?

Ver documento dedicado `estudo_ablation.md`. Resumo:

> "É a remoção sistemática de UM componente por vez, mantendo todo o
> resto constante (ceteris paribus), para isolar a contribuição desse
> componente. É o teste controlado da metodologia científica clássica
> aplicado a arquiteturas neurais. Nosso ablation é rigoroso: A/B/C
> variam apenas o mecanismo de condicionamento e o sinal, mantendo
> backbone, dataset, splits, sementes (42, 1, 2), otimizador e
> hiperparâmetros idênticos."

### 3.2 Rigor experimental (por que 3 sementes?)

**Resposta:**
> "Uma única execução pode estar em ponto de otimização atípico da
> superfície de loss. 3 sementes (42, 1, 2) permitem reportar média ±
> desvio-padrão e computar intervalo de confiança de 95 % via bootstrap
> não paramétrico. É o padrão de rigor da comunidade de fairness
> (Sagawa 2020, FSCL+ 2022, FineFACE 2024). Menos que isso não permite
> avaliar variabilidade; mais que isso não agrega estatisticamente
> além do 2× ou 3× o custo computacional."

### 3.3 Hipóteses testáveis (metodologia de Popper)

**Resposta:**
> "Cada uma das 6 hipóteses tem critério FORMAL de confirmação e de
> refutação (Cap. 3, Tabela 3). H1: κ ≥ 0.7 confirma; < 0.7 refuta.
> H3: DR menor + F1 macro ≥ baseline confirma. H6: R² ≥ 70 % confirma
> — se falhar, a tese Pangelinan ganha. A REFUTAÇÃO é resultado
> científico válido; converte-se em contribuição empírica ao debate,
> não fracasso metodológico. Esse desenho segue a lógica popperiana:
> teoria científica é aquela que se expõe à refutação."

### 3.4 Cenários A e B (por que ambos?)

**Resposta:**
> "Cenário A (raça apenas) e Cenário B (raça × gênero) atendem a duas
> exigências complementares:
> (i) Cenário A é o regime de reporte primário — 7 classes raciais do
>     FairFace, métricas multi-classe (DR, worst-class F1);
> (ii) Cenário B é a análise interseccional obrigatória desde
>      Gender Shades (Buolamwini & Gebru 2018) — usa 8 subgrupos
>      race×gender com Equal Opportunity/Equalized Odds (definições
>      formais de Hardt 2016, naturalmente binárias, aplicadas ao eixo
>      de gênero).
> Um modelo pode parecer justo entre raças e ocultar disparidade
> intra-racial entre gêneros — só a análise interseccional revela."

---

## 4. Métricas de fairness (triangulação)

### 4.1 Por que não uma métrica só?

**Resposta:**
> "Teorema da Impossibilidade (Kleinberg et al. 2017): NÃO existe
> classificador que satisfaça simultaneamente múltiplas definições
> formais de fairness quando as prevalências de base diferem entre
> grupos. Qualquer intervenção que otimize UMA definição viola outra.
> Portanto, adotar métrica única esconde escolhas éticas. Triangulamos
> DR (razão min/max), worst-class F1 (protege o pior grupo), EO/EqOdds
> (definições formais de Hardt). Cada uma expõe um trade-off diferente.
> A visualização Pareto revela quem está na fronteira do trade-off."

### 4.2 Diferença EO vs EqOdds?

**Resposta:**
> "Ambas de Hardt et al. 2016.
> - **Equal Opportunity (EO):** iguala P(Ŷ=1 | Y=1, A=a) entre grupos
>   — só a TAXA DE ACERTO condicional ao rótulo positivo.
> - **Equalized Odds (EqOdds):** iguala tanto P(Ŷ=1 | Y=1) quanto
>   P(Ŷ=1 | Y=0) — TAXA DE ACERTO e TAXA DE FALSO POSITIVO. É mais
>   estrita que EO.
> Reportamos ambas porque um modelo pode ter EO igual entre grupos e
> ainda ter FPR desigual — só EqOdds captura isso."

### 4.3 ISO/IEC 19795-10 — o que é e por que seguir?

**Resposta:**
> "Norma internacional publicada em 2024 para reporte padronizado de
> desempenho biométrico entre grupos demográficos. Define protocolo
> de estratificação, requer reporte por subgrupo, exige quantificação
> de incerteza. Aderir a essa norma dá defensabilidade internacional
> ao trabalho e alinha com a exigência do European AI Act 2024, que
> torna auditoria de fairness obrigatória para sistemas biométricos
> de alto risco."

---

## 5. Perguntas-armadilha e como responder

### "Você está treinando com o SkinToneNet?"

**Resposta:**
> "Não. O SkinToneNet (Matias 2026) não teve seus pesos publicados
> ainda. Após reunião com o orientador em agosto/2026, adotamos a
> estratégia de treinar nosso próprio classificador MST sobre datasets
> públicos (MSTE do Google, Casual Conversations v2 da Meta). Isso nos
> torna INDEPENDENTES do release externo, controla o preprocessamento
> (com camada auto-suficiente para RFW/BFW que não têm anotação MST),
> e permanece o SkinToneNet como referência externa condicional, se
> os pesos forem liberados a tempo."

### "Se a tese Pangelinan estiver certa, seu trabalho perde sentido?"

**Resposta:**
> "Não. Incorporamos essa crítica formalmente via H6 (≥ 70 % da
> variância explicada por pixel information). Se H6 for confirmada,
> temos uma REFUTAÇÃO QUANTIFICADA da hipótese de que tom de pele é o
> principal fator — resultado científico defensável, com valor
> empírico. Além disso, controlamos pixel information como confounder
> na Etapa 5 (RFW/BFW). Nossa proposta agrega valor mesmo no cenário
> de refutação parcial: primeira decomposição quantitativa
> fenotípico × algorítmico do erro Latinx (Contribuição 6)."

### "Por que 6 baselines? Não é muito?"

**Resposta:**
> "Cada baseline representa uma família metodológica distinta:
> - ResNet-34 → baseline canônico do FairFace
> - ConvNeXt-T puro → controle arquitetural moderno
> - FSCL+ → aprendizado contrastivo
> - Group DRO → otimização robusta a distribuição
> - FineFACE → atenção multi-camada Pareto-eficiente
> - Adversarial debiasing → linha inaugural
> Se cortássemos qualquer um, deixaríamos uma família fora da
> comparação. Aceitamos o custo computacional (~200-400h GPU) pela
> defensabilidade da comparação sistemática."

### "Você tem os datasets baixados?"

**Resposta (honesta):**
> "FairFace: disponível publicamente, download em preparação.
> MSTE: público, download simples (~1500 imgs).
> CCv2 Meta: público mas exige EULA — em processo.
> RFW: solicitação de acesso pendente (uso acadêmico).
> BFW: público.
> STW (Matias): não publicado ainda, requer acesso do autor.
> O código já roda em datasets sintéticos (smoke tests) — está pronto
> para consumir os datasets reais assim que baixados. A execução formal
> é planejada para Nov/2026 conforme cronograma (§5)."

---

## 6. Roteiro de estudo (próximas 4 semanas)

| Semana | Foco |
|---|---|
| Semana 1 (01-07/Set) | Reler Perez 2018 (FiLM), Dosovitskiy 2020 (ViT), Liu 2022 (ConvNeXt) integralmente |
| Semana 2 (08-14/Set) | Reler Hardt 2016, Kleinberg 2017 — treinar respostas sobre métricas |
| Semana 3 (15-21/Set) | Simulação com Prof. Quiles — perguntas cronometradas |
| Semana 4 (22-29/Set) | Ajustes finais, treino de fala, checklist logístico |

---

## 7. Sources para aprofundamento

- [ViT — Ultralytics glossary](https://www.ultralytics.com/glossary/vision-transformer-vit)
- [ViT — Roboflow blog](https://blog.roboflow.com/vision-transformers/)
- [FiLM AAAI 2018 (paper original)](https://dl.acm.org/doi/abs/10.5555/3504035.3504518)
- [FiLM Distill review — Dumoulin 2018](https://distill.pub/2018/feature-wise-transformations/)
- [FADiff — FiLM em face recognition 2025](https://www.nature.com/articles/s41598-025-34520-3)
- [Baeldung — Ablation study](https://www.baeldung.com/cs/ml-ablation-study)
- [Wikipedia — Ablation (AI)](https://en.wikipedia.org/wiki/Ablation_(artificial_intelligence))
