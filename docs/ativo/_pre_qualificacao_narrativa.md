# Pré-qualificação narrativa — estrutura de texto

> **Versão**: 1.0 — primeira escrita pré-qualificação.
> **Data**: 2026-06-10.
> **Propósito**: estrutura narrativa fechada com objetivo claro,
> servir de base para conversão em LaTeX (Overleaf) na próxima
> etapa.
> **Storytelling aprovado** pelo orientador (Prof. Marcos Quiles):
> Contexto → Problemas → Estado-da-arte → Gap → Objetivo → Como
> será feito.

---

## 1. Contexto

Sistemas de reconhecimento facial deixaram de ser uma tecnologia
restrita a laboratórios para se tornar parte cotidiana da
infraestrutura social. Aplicações vão de **catracas biométricas em
academias e empresas**, passando por **desbloqueio de aparelhos
pessoais**, **autenticação bancária e identidade digital**, até
**controle de fronteira em aeroportos** e **identificação policial
em vias públicas**. Em escala global, o National Institute of
Standards and Technology auditou em 2019 cento e oitenta e nove
algoritmos de reconhecimento facial sobre dezoito milhões de
imagens de oito milhões e meio de pessoas, na maior auditoria
pública já realizada em biometria facial [Grother, Ngan & Hanaoka,
2019, NISTIR 8280].

A literatura científica acompanhou essa expansão. Modelos modernos
combinam grandes datasets faciais (FairFace, VGGFace2, MS-Celeb-1M)
com arquiteturas neurais profundas (ResNet, ConvNeXt, Vision
Transformers) e técnicas avançadas de loss (ArcFace, CosFace),
atingindo desempenhos próximos ao humano em condições controladas.
Em paralelo, vision-language models como CLIP [Radford et al.,
2021] e PaliGemma criaram nova categoria de classificadores
faciais via reformulação como Visual Question Answering — o estado
da arte atual em classificação racial multi-classe sobre FairFace é
**FaceScanPaliGemma**, com 75,7% de acurácia e F1 macro de 75%
[AlDahoul, Tan, Kasireddy & Zaki, 2026, Nature Scientific Reports].

Esta dissertação se insere nesta interseção entre maturidade
tecnológica e responsabilidade social. O reconhecimento facial
funciona para a maioria das pessoas — mas a documentação científica
mostra que não funciona igualmente bem para todas.

---

## 2. Problemas existentes

O problema central que motiva esta dissertação é simples de
enunciar e tecnicamente difícil de resolver: **sistemas de
reconhecimento facial apresentam disparidade demográfica
significativa em accuracy entre grupos raciais, mesmo quando
treinados em datasets explicitamente balanceados**.

### 2.1 Disparidade documentada no estado-da-arte

O estado-da-arte atual em classificação racial em sete classes
sobre o FairFace, o FaceScanPaliGemma [AlDahoul et al., 2026],
atinge F1 macro de setenta e cinco por cento. Mas o detalhamento
por classe racial revela disparidade severa: F1 de noventa por cento
para faces Black, oitenta por cento para White, sessenta por cento
para Latinx/Hispanic, e sessenta e sete por cento para Southeast
Asian. **A pior classe (Latinx) tem F1 trinta pontos percentuais
abaixo da melhor classe (Black)**.

Esta disparidade não é artefato de um único modelo. O baseline
ResNet-34 originalmente treinado por Kärkkäinen & Joo [2021] sobre
o FairFace mostra padrão semelhante: setenta e dois por cento de
accuracy geral, mas apenas cinquenta e nove vírgula seis por cento
em Hispanic versus oitenta e três vírgula dois por cento em Black
[Lin, Kim & Joo, 2022]. A hierarquia de dificuldade entre raças é
**estável across métodos**, sugerindo um problema estrutural, não
algorítmico contingente.

### 2.2 Balanceamento de dados não resolve

A solução intuitiva — balancear o dataset por raça — é
insuficiente. O próprio FairFace foi construído com este propósito
[Kärkkäinen & Joo, 2021], distribuindo as cento e oito mil
quinhentas e uma imagens aproximadamente igual entre sete
categorias raciais. Mesmo assim, a disparidade Latinx versus White
persiste.

Kolla & Savadamuthu [2022] conduziram dezesseis experimentos com
proporções variáveis de raças no treino, concluindo explicitamente
que "distribuição uniforme de raças nos dados de treino não
garante face recognition livre de viés". Wang et al. [2019], ao
introduzir o dataset RFW (Racial Faces in-the-Wild), observaram que
"algumas raças permanecem intrinsecamente mais difíceis de
reconhecer, mesmo com dados de treinamento balanceados". Em escala
industrial, o relatório NISTIR 8280 documentou diferenciais de
falso positivo de dez a cem vezes maiores entre grupos raciais em
algoritmos comerciais [Grother, Ngan & Hanaoka, 2019].

### 2.3 Implicações sociais

A disparidade tem consequências concretas. Quando um sistema de
reconhecimento facial é utilizado em catraca biométrica, autenticação
financeira ou identificação policial, **a probabilidade de erro
depende do rosto da pessoa**. Casos documentados de prisão errada
por falso match em sistemas comerciais de FR já levaram a litígios
e moratórias regulatórias. O European AI Act de 2024 incluiu
auditoria de fairness como requisito formal para sistemas de IA
classificados como alto risco [Lafargue, Claeys & Loubes, 2025].

A questão científica deixou de ser "existe viés?" — está
documentado há quase uma década, desde o estudo seminal Gender
Shades de Buolamwini & Gebru [2018]. A questão atual é: **qual é
o mecanismo gerador desse viés residual mesmo em datasets
balanceados, e como mitigá-lo de forma eficaz?**

---

## 3. O que tem sido feito (estado-da-arte)

A literatura científica de fairness em sistemas faciais tem atacado
o problema por seis frentes principais. Apresenta-se a seguir um
panorama estruturado, baseado no corpus de trinta e sete artigos
catalogados desta dissertação (Rodadas 1 a 6 de triagem
bibliográfica).

### 3.1 Datasets balanceados como mitigação de primeiro estágio

A primeira geração de respostas ao problema foram **datasets
balanceados por design**. O FairFace [Kärkkäinen & Joo, 2021]
balanceia sete categorias raciais — White, Black, Indian, East
Asian, Southeast Asian, Middle Eastern, Latinx/Hispanic. O Racial
Faces in-the-Wild [Wang et al., 2019] balanceia quatro categorias
em verificação 1:1. O Balanced Faces in the Wild [Robinson et al.,
2020] cobre oito subgrupos race × gender. Mais recentemente, o
Casual Conversations [Hazirbas et al., 2021] introduziu
self-reported gender e age com Fitzpatrick skin tone anotado por
treinados.

A persistência da disparidade Latinx no FairFace, apesar do
balanceamento, demonstrou que esta primeira geração não foi
suficiente.

### 3.2 Mitigação algorítmica via loss functions e arquiteturas

Em paralelo aos datasets, surgiu uma família de técnicas de
mitigação algorítmica. Park et al. [2022] propuseram Fair
Supervised Contrastive Learning, demonstrando formalmente que
SupCon padrão incentiva encoders a aprenderem informação
demográfica em datasets enviesados, e propondo restringir negativos
ao mesmo grupo sensível. Sagawa et al. [2020] formalizaram
distributionally robust optimization sobre grupos (Group DRO),
demonstrando que regularização forte é essencial para que o
método supere baselines empirical risk minimization. Manzoor &
Rattani [2024] propuseram cross-layer mutual attention learning
(FineFACE), atingindo operação Pareto-eficiente em CelebA — melhora
simultânea de fairness e accuracy.

Mais recentemente, Liu et al. [2025] em ACM FAccT propuseram
Bayesian Network-informed Meta Reweighting, introduzindo a noção
de face component fairness como surrogate para demographic
fairness. Lin, Kim & Joo [2022] introduziram FairGRAPE, técnica de
pruning fairness-aware que preserva representação per-grupo, e
documentaram independentemente o baseline canônico de setenta e
dois por cento accuracy ResNet-34 sobre FairFace race 7-class
in-domain.

### 3.3 Vision-language models como novo paradigma de classificação

Uma terceira frente emergiu com vision-language models. AlDahoul
et al. [2024, 2026] reformularam classificação racial como Visual
Question Answering sobre PaliGemma fine-tuned, atingindo o atual
estado-da-arte (FaceScanPaliGemma, 75,7% accuracy, 75% F1 macro).
Para fairness em VLMs especificamente, Luo et al. [2024] em CVPR
propuseram FairCLIP, primeiro fair vision-language dataset médico,
introduzindo abordagem baseada em distância de Sinkhorn entre
distribuições demográficas.

VLMs trouxeram desempenho superior mas também novas formas de
viés. Vision-language models são quatro a sete vezes mais propensos
a classificar incorretamente indivíduos com pele escura [GRAS
Benchmark, 2025]. CLIP especificamente apresenta taxa de
classificação errada de quatorze por cento em faces Black versus
menos de oito por cento em outros grupos.

### 3.4 Skin tone como dimensão alternativa à raça

Desde o estudo Gender Shades de Buolamwini & Gebru [2018], tom de
pele tem sido proposto como **dimensão alternativa de auditoria
de fairness**, mais estável conceitualmente que categoria racial.
Fitzpatrick [1988] originalmente desenvolveu escala de seis tipos
para fototerapia dermatológica, posteriormente adotada em fairness
research apesar de seu viés caucasiano-cêntrico — três dos seis
tipos cobrem o espectro perceived as White.

Em 2023, Schumann et al. [NeurIPS] estabeleceram a **Monk Skin Tone
Scale**, escala de dez pontos desenvolvida em parceria entre Ellis
Monk (sociólogo, Harvard) e Google, designed especificamente para
auditoria de fairness em IA. Em 2026, Pereira Matias, Costa, Neto
& Novello de Brito [arXiv:2603.02475] introduziram o **SkinToneNet**
— Vision Transformer Small fine-tuned em STW (Skin Tone in the
Wild, quarenta e dois mil trezentos e treze imagens de três mil
quinhentos e sessenta e quatro indivíduos), atingindo SOTA em
classificação MST cross-domain.

### 3.5 Fair representation learning e teoria da transferência

Uma linhagem teórica importante foi inaugurada por Zemel et al.
[2013, ICML, Test-of-Time Award 2023], com o paradigma de
**Learning Fair Representations**: aprender mapeamento intermediário
que preserva utilidade para a tarefa-alvo enquanto ofusca informação
sobre atributo protegido. Madras et al. [2018, ICML] estenderam o
paradigma com LAFTR — Learning Adversarially Fair and Transferable
Representations — provando o **Teorema 1 de transferência**: se a
representação Z é fair em uma tarefa, qualquer classificador
downstream sobre Z herda um limite superior de violação de
fairness.

Aguirre & Dredze [2023] demonstraram empíricamente, em domínio NLP,
que objetivos de fairness demográfico transferem entre tarefas em
framework multi-task, com reduções de quinze a quarenta e quatro
por cento em ε-DEO mantendo F1. Hardt, Price & Srebro [2016,
NeurIPS] estabeleceram as métricas formais Equal Opportunity e
Equalized Odds que toda esta literatura subsequente adota como
referência.

### 3.6 Auditoria de datasets e mecanismos teóricos

Em paralelo, Dominguez-Catena, Paternain & Galar [2024, Information
Fusion] propuseram DSAP — framework unificado de auditoria de
datasets via Renkonen similarity, importado da ecologia. Lafargue,
Claeys & Loubes [2025, ECML PKDD] propuseram pipeline de auditoria
com testes estatísticos uncertainty-aware. Dooley et al. [2022]
demonstraram via Neural Architecture Search que **biases são
inerentes a arquiteturas neurais**, encontrando suíte de modelos
Pareto-superior em accuracy e fairness simultaneamente.

Quanto a mecanismos arquiteturais de conditioning, Perez et al.
[2018, AAAI] propuseram FiLM — Feature-wise Linear Modulation —
camada de modulação afim por canal aprendida a partir de contexto
auxiliar. FiLM é parameter-efficient (cerca de um por cento
overhead) e amplamente adotado em VQA e geração condicional, mas
**nunca foi testado em fairness facial até a presente dissertação**.

---

## 4. O que falta ser explorado (gap)

A análise sistemática do corpus revela **cinco gaps científicos**
não cobertos pela literatura atual.

### 4.1 Gap 1: matriz pública MST × race classes do FairFace

Apesar da disponibilidade do MST como escala padrão moderna desde
2023 [Schumann et al.], e apesar da existência de classificador MST
SOTA pré-treinado [Pereira et al., 2026], **nenhum trabalho publicou
a distribuição cruzada entre as dez classes MST e as sete categorias
raciais do FairFace**. Pereira et al. auditam oito datasets faciais
(incluindo FairFace) reportando apenas distribuições agregadas —
"FairFace exibe ausência de MST classes 6-10" — sem cross-tabulation
per-race. A pergunta "como Latinx se distribui em MST versus como
East Asian se distribui?" permanece **sem resposta pública**.

### 4.2 Gap 2: skin tone como sinal condicionante arquitetural em race classification multi-classe

A literatura de mitigação algorítmica testa três abordagens
principais — contrastive learning (FSCL), distributionally robust
optimization (Group DRO) e adversarial debiasing (Zhang 2018,
LAFTR). **Nenhuma testa explicitamente o uso de tom de pele como
sinal auxiliar condicionante de uma rede de classificação racial via
mecanismo arquitetural**. FiLM [Perez et al., 2018] fornece o
mecanismo formal, mas nunca foi aplicado a fairness facial. A
combinação SkinToneNet + FiLM + race classifier é, portanto, **gap
metodológico identificável**.

### 4.3 Gap 3: decomposição fenotípico (irredutível) vs algorítmico (redutível)

A disparidade Latinx no FairFace, persistente across métodos,
sugere componente estrutural relacionado a sobreposição fenotípica
entre categorias raciais. Lewontin [1972] demonstrou geneticamente
que oitenta e cinco por cento da variação humana é intra-populacional
— "raça" captura apenas uma fração minoritária da diversidade. A
AAPA Statement de Fuentes et al. [2019] consolidou institucionalmente
que "race does not provide an accurate representation of human
biological variation". Mas **a literatura computacional não decompõe
o erro de classificação racial em componente fenotípico irredutível
versus componente algorítmico mitigável**. Esta decomposição é
inédita e seria diagnóstico relevante para a discussão ética sobre
limites estruturais de race classification.

### 4.4 Gap 4: fair transferência empírica de classification para face recognition em CV facial

Madras et al. [2018] provaram teoricamente que fairness transfere
entre tarefas via shared representation. Aguirre & Dredze [2023]
demonstraram empíricamente em domínio NLP. **Não existe demonstração
empírica equivalente em computer vision facial** — i.e., treinar
backbone fair em race classification e medir se a propriedade fair
herda em face recognition downstream (RFW, BFW). Pangelinan et al.
[2023] introduzem uma ressalva importante (pixel info pode dominar
disparity em FR), mas a demonstração empírica direta da
transferência permanece em aberto.

### 4.5 Gap 5: triangulação de métricas para race classification multi-classe

A literatura adota Equal Opportunity e Equalized Odds [Hardt et al.,
2016] como métricas canônicas, mas estas são originalmente binárias
(Y, A ∈ {0,1}). Para race classification em sete classes, as
adaptações são fragmentadas: Kärkkäinen & Joo [2021] usam ε
log-ratio max disparity; Lin et al. [2022] usam ρ(A) desvio padrão
de accuracy entre grupos; AlDahoul et al. [2026] reportam apenas F1
por classe. **Não há protocolo consensual de triangulação de métricas
multi-classe** que satisfaça simultaneamente: (a) interpretabilidade
operacional, (b) sensibilidade ao pior subgrupo, (c) capacidade de
captar disparidade intersectional race × gender. A combinação **DR +
worst-class F1 + EO_h/EOD por classe** proposta nesta dissertação
endereça este gap, fundamentada no Teorema da Impossibilidade de
Kleinberg, Mullainathan & Raghavan [2017].

---

## 5. Objetivo

### 5.1 Objetivo geral

> **Esta dissertação tem como objetivo desenvolver e avaliar um
> pipeline de classificação racial em imagens faciais que incorpora
> tom de pele (escala Monk Skin Tone, 10 pontos) como sinal auxiliar
> condicionante via mecanismo arquitetural, com o propósito de
> mitigar viés racial demonstrável no estado-da-arte atual —
> particularmente a disparidade severa entre classes raciais bem
> representadas (Black, F1 aproximadamente 90%) e classes
> sub-representadas (Latinx/Hispanic, F1 aproximadamente 60%)
> documentada em modelos como FaceScanPaliGemma sobre o dataset
> FairFace. A contribuição principal é a primeira instância
> empírica documentada do uso de tom de pele explícito como
> contexto arquitetural para race classification multi-classe, com
> avaliação rigorosa via triangulação de métricas multi-classe e
> demonstração de fair transferência do mecanismo para downstream
> face recognition.**

### 5.2 Objetivos específicos

| # | Objetivo | Capítulo | Hipóteses |
|---|---|---|---|
| **OE1** | Quantificar a distribuição cruzada Monk Skin Tone × race classes sobre o FairFace via SkinToneNet pré-treinado, com validação humana em subset, entregando a primeira matriz pública desta distribuição | Cap. 1 | H3 |
| **OE2** | Conduzir avaliação metodológica de modelos pré-treinados disponíveis para classificação MST e justificar a escolha do modelo adotado | Cap. 1 | (metodológico) |
| **OE3** | Implementar e avaliar pipeline SkinToneNet → ConvNeXt-T + conditioning → race classifier sobre FairFace, comparando contra seis baselines de mitigação, com ablation arquitetural FiLM versus CLIP-conditioning | Cap. 2 | H1, H2, H4 |
| **OE4** | Demonstrar empíricamente que o backbone fair-treinado no Cap. 2 transfere a propriedade fair para tarefa downstream de face recognition (RFW ou BFW), com controle explícito de pixel information como confounder | Cap. 3 | H5, H6 |
| **OE5** | Decompor o erro de classificação racial Latinx em componente fenotípico (irredutível pelo overlap MST) versus componente algorítmico (mitigável), quantificando a fronteira estrutural de classificação multi-classe | Cap. 4 | (síntese) |

### 5.3 Hipóteses de pesquisa

| ID | Hipótese | Confirmação | Refutação |
|---|---|---|---|
| **H1** | Pipeline com tom de pele como contexto supera baseline ResNet-34 em F1 macro ≥ +2pp e reduz Disparity Ratio ≥ 20% | Ambos simultâneos | Qualquer um abaixo |
| **H2** | ConvNeXt-T sem conditioning ganha +2 a +5pp F1 sobre ResNet-34, mas Latinx permanece em ~60% (±3pp) | Ganho no range E Latinx invariante | Fora do range OU Latinx muda |
| **H3** | Latinx exibe spread MST em ≥ 5 das 10 classes, com sobreposição forte com White, Middle East e Indian | Spread ≥ 5 com pico distribuído | Spread < 5 |
| **H4** | ≥ 50% das misclassificações Latinx→outras classes do baseline estão em zonas MST de sobreposição | ≥ 50% em overlap | < 50% (sugere erro algorítmico, não fenotípico) |
| **H5** | Condicionamento por MST melhora fairness em face recognition downstream, **quando pixel information é controlada via face crop e alinhamento** | Black/African ≥ +3pp em RFW/BFW | < +3pp ou degradação |
| **H6** | Disparity residual em FR após conditioning é predominantemente explicada por diferenças de pixel information [Pangelinan et al., 2023], não por skin tone | ≥ 70% variance explicada por pixel info | < 70% (sugere viés algorítmico residual) |

### 5.4 Contribuições científicas previstas

| C# | Contribuição | Diferenciação na literatura |
|---|---|---|
| **C1** | Avaliação metodológica comparativa de modelos pré-treinados de skin tone classification, com protocolo de escolha justificado | Sem benchmark unificado público entre SkinToneNet, Casual Conversations baseline, Google API e alternativas |
| **C2** | Matriz pública Monk Skin Tone × race classes sobre o FairFace validation set | Pereira et al. [2026] auditam mas não cruzam por raça; gap inédito |
| **C3** | Primeira aplicação documentada de FiLM-conditioning [Perez et al., 2018] a fairness facial em race classification multi-classe | FiLM nunca testado em fairness; combinação tom de pele × raça inédita |
| **C4** | Triangulação de métricas multi-classe — Disparity Ratio + worst-class F1 + Equal Opportunity por classe + Equalized Odds por classe — fundamentada no Teorema da Impossibilidade [Kleinberg et al., 2017] | Adapta Hardt [2016] para multi-classe com ablation race × gender |
| **C5** | Demonstração empírica de fair transferência classification → face recognition em CV facial | LAFTR é teórico; Aguirre & Dredze [2023] é NLP; CV facial é original |
| **C6** | Decomposição quantitativa do erro Latinx em componente fenotípico irredutível versus algorítmico mitigável | Diagnóstico inédito; combina C1 + C2 + C5 |
| **C7** | Comparativo arquitetural FiLM-conditioning versus CLIP-conditioning para fairness facial | Endereça lacuna na literatura sobre mecanismos modernos de conditioning aplicados a fairness |

---

## 6. Como será feito (metodologia resumida)

### 6.1 Pipeline experimental em seis etapas

O pipeline metodológico aprovado pelo orientador estrutura-se em
seis etapas sequenciais:

1. **Etapa 1 — Classificador de tom de pele MST**: usar SkinToneNet
   pré-treinado [Pereira et al., 2026] como insumo, mediante
   validação independente em subset FairFace anotado manualmente.
2. **Etapa 2 — Auditoria FairFace**: aplicar SkinToneNet sobre
   FairFace validation set e construir a matriz pública MST × race
   classes (Contribuição C2).
3. **Etapa 3 — Race classifier com conditioning**: treinar
   ConvNeXt-T fine-tuned em FairFace train, com camadas FiLM por
   estágio recebendo vetor MST como contexto.
4. **Etapa 4 — Comparação contra baselines**: avaliar o pipeline
   contra seis baselines de mitigação (ResNet-34, ConvNeXt-T puro,
   FSCL+, Group DRO, FineFACE, Adversarial debiasing), reportando
   triangulação completa de métricas (Contribuição C4).
5. **Etapa 5 — Fair transferência**: aplicar o backbone fair em
   face recognition downstream (RFW ou BFW), com controle de pixel
   information.
6. **Etapa 6 — Síntese decompositiva**: combinar resultados de C2
   e C5 para quantificar componente fenotípico versus algorítmico
   do erro Latinx (Contribuição C6).

### 6.2 Estrutura da dissertação

| Capítulo | Conteúdo | Hipóteses testadas |
|---|---|---|
| **Cap. 1** | Auditoria fenotípica do FairFace: matriz MST × race + avaliação de modelos pré-treinados MST | OE1, OE2, H3 |
| **Cap. 2** | Pipeline condicionado de race classification: SkinToneNet + ConvNeXt-T + FiLM, com ablation FiLM vs CLIP | OE3, H1, H2, H4 |
| **Cap. 3** | Fair transferência para face recognition: pipeline em RFW/BFW com controle pixel info | OE4, H5, H6 |
| **Cap. 4** | Síntese decompositiva e discussão ética | OE5, C6 |

### 6.3 Protocolo de avaliação

**Datasets**:
- Treino: FairFace train split (86.744 imgs).
- Validação MST: STW [Pereira et al., 2026] + Casual Conversations
  [Hazirbas et al., 2021] em subset.
- Teste race classification: FairFace test split (10.954 imgs).
- Teste fair transfer: RFW e/ou BFW.

**Baselines**:
- ResNet-34 (canônico FairFace).
- ConvNeXt-T puro (controle arquitetural).
- FSCL+ [Park et al., 2022].
- Group DRO [Sagawa et al., 2020].
- FineFACE [Manzoor & Rattani, 2024].
- Adversarial debiasing [Zhang, Lemoine & Mitchell, 2018].

**Métricas (triangulação Contribuição C4)**:
- Protocolo principal (race apenas): F1 macro + Disparity Ratio +
  worst-class F1.
- Ablation intersectional (race × gender): Equal Opportunity por
  classe + Equalized Odds por classe.

**Rigor estatístico**:
- Três seeds independentes (42, 1, 2) com média ± desvio padrão.
- Comparação pareada contra baselines.
- Intervalos de confiança via bootstrap.

### 6.4 Cronograma de execução

O cronograma de execução da pesquisa abrange aproximadamente vinte
e oito semanas após a aprovação da qualificação:

| Fase | Duração estimada | Entrega |
|---|---|---|
| Setup metodológico | 2 semanas | Documentos com especificações |
| Cap. 1 — Auditoria MST | 4 semanas | Resultado de H3 + protocolo de escolha do modelo |
| Cap. 2 — Pipeline + ablation | 10-12 semanas | Resultados de H1, H2, H4 + C3, C7 |
| Cap. 3 — Fair transferência | 6 semanas | Resultados de H5, H6 |
| Síntese | 4 semanas | Decomposição C6 |
| Escrita final | 8-12 semanas (paralelo) | Texto final da dissertação |
| **Defesa prevista** | **Janeiro a Março de 2027** | — |

---

## 7. Resumo

Esta dissertação propõe operacionalizar uma intuição que a literatura
de fairness facial reconhece mas raramente formaliza: **tom de pele
e categoria racial são variáveis distintas, e a primeira pode servir
como sinal auxiliar arquitetural para mitigar viés na segunda**.
A demonstração empírica desta operacionalização — via pipeline
SkinToneNet + ConvNeXt-T + FiLM-conditioning sobre o FairFace — é
**inédita**, e a triangulação de métricas + decomposição
fenotípico/algorítmico que a sustentam constituem contribuições
metodológicas adicionais.

O resultado esperado não é apenas redução numérica da disparidade —
é também **diagnóstico estrutural**: quantificar quanto do erro de
classificação racial é matematicamente inevitável (overlap fenotípico
entre categorias) e quanto é algoritmicamente mitigável. Esse
diagnóstico é relevante para a discussão ética sobre os limites
estruturais de race classification multi-classe, tema que extrapola
a engenharia e dialoga com a posição de antropologia biológica
[Fuentes et al., 2019] e a evidência genética clássica [Lewontin,
1972].
