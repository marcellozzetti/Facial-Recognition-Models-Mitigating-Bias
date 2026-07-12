---
tipo: projeto-pesquisa
finalidade: anexo a Declaracao de Responsabilidade (Art. 8 Res. 200/2021/CONSU)
data: 2026-07-08
status: preparacao
---

# Projeto de Pesquisa — Anexo à Declaração de Responsabilidade

> Documento formal do projeto para submissão junto à Declaração de
> Responsabilidade prevista no parágrafo único do Art. 8º da Resolução
> nº 200/2021/CONSELHO UNIVERSITÁRIO da Universidade Federal de São Paulo.

## 1. Identificação

| Campo | Valor |
|---|---|
| **Título do projeto** | Equidade Racial em Classificação Facial: Pipeline Condicionado por Tom de Pele (Escala Monk Skin Tone) via FiLM sobre o Dataset FairFace |
| **Mestrando** | Marcello Ozzetti |
| **Orientador / Pesquisador Responsável** | Prof. Dr. Marcos Quiles |
| **Programa** | Mestrado em Ciência da Computação |
| **Instituição** | Universidade Federal de São Paulo — Instituto de Ciência e Tecnologia (Unifesp / ICT) |
| **Linha de pesquisa** | Visão Computacional e Equidade Algorítmica |
| **Período previsto** | julho de 2026 a outubro de 2026 (qualificação) |

## 2. Resumo

Sistemas modernos de reconhecimento facial apresentam disparidade
demográfica significativa entre grupos raciais, mesmo quando treinados
sobre datasets explicitamente balanceados. No estado da arte atual,
o modelo FaceScanPaliGemma sobre o dataset público FairFace atinge
F1 de 90 % para faces classificadas como Black, mas apenas 60 % para
faces classificadas como Latinx — uma diferença de trinta pontos
percentuais que persiste através de múltiplos métodos e arquiteturas.
Esta dissertação propõe um pipeline de classificação racial que
incorpora tom de pele (escala Monk Skin Tone, dez classes) como sinal
auxiliar condicionante via mecanismo arquitetural FiLM (Feature-wise
Linear Modulation), e avalia sua capacidade de reduzir essa
disparidade sem sacrificar acurácia agregada. A pesquisa é
inteiramente computacional, conduzida sobre datasets públicos
secundários, sem coleta primária, intervenção ou qualquer forma de
envolvimento direto ou indireto de seres humanos.

## 3. Justificativa e problema de pesquisa

O reconhecimento facial deixou de ser tecnologia laboratorial e passou
a integrar a infraestrutura social — desbloqueio de dispositivos,
autenticação bancária, identidade digital, controle de fronteiras e
identificação policial. Em escala industrial, o National Institute
of Standards and Technology auditou em 2019 cento e oitenta e nove
algoritmos de reconhecimento facial sobre dezoito milhões de imagens,
documentando diferenciais de dez a cem vezes na taxa de falso
positivo entre grupos raciais.

Apesar de mais de uma década de pesquisa em equidade algorítmica,
a disparidade persiste. O baseline ResNet-34 sobre FairFace, o
FaceScanPaliGemma via VLM, e mesmo modelos com balanceamento
explícito de dados continuam apresentando F1 aproximadamente 60 %
para a categoria Latinx / Hispanic. Trabalhos recentes sugerem que
essa disparidade não é apenas artefato algorítmico, mas reflete
heterogeneidade fenotípica intra-categorial que a rotulagem racial
monolítica do FairFace não captura adequadamente.

O problema de pesquisa desta dissertação é, portanto: **condicionar
arquiteturalmente um classificador racial pelo tom de pele reduz a
disparidade demográfica sem sacrificar acurácia agregada, e permite
decompor o erro em componentes fenotípico e algorítmico?**

## 4. Objetivos

### 4.1 Objetivo geral

Desenvolver e avaliar um pipeline de classificação racial em imagens
faciais que incorpora tom de pele (escala Monk Skin Tone) como sinal
auxiliar condicionante via mecanismo arquitetural FiLM, com o
propósito de mitigar viés racial demonstrável no estado da arte
atual — particularmente a disparidade severa entre classes raciais
bem representadas e classes sub-representadas documentada em modelos
como FaceScanPaliGemma sobre o dataset FairFace.

### 4.2 Objetivos específicos

1. **Auditoria fenotípica do FairFace** — quantificar a distribuição
   cruzada MST × classes raciais sobre o FairFace via SkinToneNet
   pré-treinado, entregando a primeira matriz pública dessa distribuição.
2. **Avaliação metodológica de classificadores MST** — comparar
   SkinToneNet, Casual Conversations baseline e alternativas
   disponíveis, com sensitivity analysis a dois ou três classificadores
   alternativos para mitigar risco de propagação de viés.
3. **Race classifier com conditioning arquitetural** — treinar
   ConvNeXt-T fine-tuned em FairFace, com camadas FiLM por estágio
   recebendo o vetor MST como contexto.
4. **Comparação sistemática contra baselines de mitigação** — avaliar
   o pipeline contra seis baselines (ResNet-34, ConvNeXt-T puro,
   FSCL+, Group DRO, FineFACE, Adversarial debiasing) via
   triangulação de métricas (DR + worst-class F1 + EO_h/EOD).
5. **Fair transferência para face recognition downstream** — aplicar
   o backbone fair em face recognition em RFW ou BFW, com controle
   explícito de pixel information como confounder.
6. **Síntese decompositiva** — combinar resultados dos objetivos
   1 e 5 para quantificar componente fenotípico (irredutível) versus
   componente algorítmico (mitigável) do erro Latinx.

## 5. Metodologia

### 5.1 Natureza da pesquisa

A pesquisa é **puramente computacional**. Consiste em:

- Treinamento e avaliação de modelos de aprendizado profundo sobre
  datasets de imagens faciais já existentes e publicamente disponíveis.
- Análise estatística de métricas de desempenho estratificadas por
  grupo demográfico.
- Comparação sistemática entre configurações arquiteturais.
- Síntese analítica dos resultados.

### 5.2 Datasets utilizados — todos secundários e públicos

| Dataset | Origem | Licença | Uso na pesquisa |
|---|---|---|---|
| FairFace | Kärkkäinen & Joo (2021), UCLA | CC BY 4.0 (uso acadêmico) | Dataset principal — 108.501 imagens balanceadas em 7 categorias raciais |
| RFW — Racial Faces in the Wild | Wang et al. (2019), BUPT | Research-only (acesso mediante solicitação acadêmica) | Avaliação de fair transferência downstream |
| BFW — Balanced Faces in the Wild | Robinson et al. (2020), RIT | CC BY 4.0 | Avaliação alternativa de fair transferência |
| BUPT-Balancedface | Wang, Zhang, Deng (2022), BUPT | Research-only | Sensitivity analysis do backbone |
| STW — Skin Tone in the Wild | Pereira et al. (2026) | Research-only | Base de treinamento do SkinToneNet (usado apenas como modelo pré-treinado) |

**Todas as imagens desses datasets foram coletadas, anotadas e
disponibilizadas por seus autores originais anteriormente ao início
desta pesquisa. Este projeto não coleta, não anota e não distribui
qualquer imagem original.**

### 5.3 Pipeline metodológico em seis etapas

1. **Etapa 1 — Classificador MST**: uso do SkinToneNet pré-treinado
   (Pereira 2026) como insumo, mediante validação de concordância
   interna em subset estratificado.
2. **Etapa 2 — Auditoria FairFace**: aplicação do SkinToneNet sobre
   o FairFace validation set e construção da matriz pública MST ×
   raça.
3. **Etapa 3 — Race classifier com conditioning**: treinamento do
   ConvNeXt-T em FairFace train, com camadas FiLM recebendo o vetor
   MST como contexto arquitetural.
4. **Etapa 4 — Comparação contra baselines**: avaliação do pipeline
   contra seis baselines de mitigação, com triangulação métrica.
5. **Etapa 5 — Fair transferência**: aplicação do backbone fair em
   face recognition downstream (RFW ou BFW).
6. **Etapa 6 — Síntese decompositiva**: quantificação do componente
   fenotípico versus algorítmico do erro Latinx.

### 5.4 Rigor experimental

Todos os experimentos serão conduzidos com múltiplas sementes
aleatórias (mínimo três: 42, 1, 2), comparação pareada entre
configurações, intervalos de confiança via bootstrap e reporte
estratificado por raça e por interseção raça × gênero. As métricas
seguirão triangulação (DR + worst-class F1 + EO_h/EOD) para
endereçar o Teorema da Impossibilidade de Kleinberg.

### 5.5 Validação humana — modalidade interna

Para o Objetivo Específico 1, será conduzida validação de
concordância entre o SkinToneNet e anotação manual em subset
estratificado de aproximadamente duzentas a trezentas imagens do
FairFace. **Esta validação será realizada exclusivamente pela equipe
acadêmica do projeto** — Mestrando e Orientador (e Coorientador,
se designado). Não haverá contratação de anotadores externos, não
haverá uso de plataformas de crowdsourcing, e não haverá coleta de
qualquer dado pessoal de indivíduos externos à equipe.

## 6. Aspectos éticos — enquadramento no Art. 8º da Resolução 200/2021

Este projeto **não envolve, direta ou indiretamente, seres humanos
como sujeitos de pesquisa**, conforme a definição estabelecida pela
Resolução nº 200/2021 do Conselho Universitário da Universidade
Federal de São Paulo e pelas Resoluções nº 466/2012 e nº 510/2016
do Conselho Nacional de Saúde. Justifica-se:

### 6.1 Ausência de coleta primária

O projeto **não coleta**, em nenhuma etapa, imagens de rostos,
dados biométricos, dados demográficos, opiniões, comportamento ou
qualquer outra informação de seres humanos identificáveis ou
identificados. Todas as imagens utilizadas são secundárias,
provenientes de datasets construídos e disponibilizados
publicamente por terceiros anteriormente ao início desta pesquisa.

### 6.2 Ausência de intervenção

O projeto **não realiza intervenção** de qualquer natureza sobre
seres humanos — não há entrevistas, questionários, experimentos,
testes psicológicos, exames físicos, uso de dispositivos, alteração
de ambiente ou qualquer procedimento que produza efeito sobre
pessoas físicas.

### 6.3 Ausência de envolvimento indireto via crowdsourcing

Diferentemente de projetos que empregam plataformas de crowdsourcing
(Amazon Mechanical Turk, Prolific, Toloka, entre outras) para
anotação de dados, este projeto **não contrata anotadores externos**.
A validação manual prevista no Objetivo Específico 1 será conduzida
exclusivamente por membros da equipe acadêmica do projeto,
enquadrando-se como atividade de pesquisa científica interna e não
como recrutamento de participantes.

### 6.4 Ausência de identificação de sujeitos

As análises produzidas nesta pesquisa referem-se a **grupos
demográficos agregados** (categorias raciais, tons de pele) e não a
indivíduos identificáveis. Nenhum resultado publicado permitirá
identificar pessoas específicas presentes nos datasets utilizados.

### 6.5 Ausência de uso de animais vertebrados vivos

O projeto **não utiliza animais vertebrados vivos** de qualquer
natureza, não se enquadrando na competência da Comissão de Ética
no Uso de Animais (CEUA).

### 6.6 Enquadramento formal

Pelo exposto, este projeto se enquadra no **Art. 8º da Resolução
nº 200/2021/CONSU** — "Os projetos de pesquisa que não envolvem
seres humanos, direta e indiretamente, nem animais vertebrados
vivos, estão dispensados de cadastro" — e é objeto da Declaração
de Responsabilidade prevista no parágrafo único do referido artigo,
assinada pelo estudante, pelo orientador e pelo chefe do
Departamento ao qual o orientador está vinculado.

## 7. Cronograma

| Marco | Data | Descrição |
|---|---|---|
| Entrega da primeira revisão da qualificação ao orientador | 15/jul/2026 | Documento completo em LaTeX / Overleaf |
| Pedido formal de qualificação ao Programa | 30/jul/2026 | Prazo regimental do PPG-CC / ICT |
| Defesa da qualificação | outubro/2026 | Sujeita à prorrogação já solicitada de dois meses |
| Experimentos completos e escrita da dissertação | novembro/2026 a maio/2027 | — |
| Defesa da dissertação | segundo semestre/2027 | — |

## 8. Referências principais

**Datasets e benchmarks**:
Kärkkäinen & Joo (2021). *FairFace: Face Attribute Dataset*. WACV.
Wang et al. (2019). *Racial Faces in the Wild — RFW*. ICCV.
Robinson et al. (2020). *Face Recognition: Too Bias, or Not Too Bias?*. CVPRW.
Hazirbas et al. (2021). *Casual Conversations*. CVPRW.

**Fundamentação teórica em fairness**:
Buolamwini & Gebru (2018). *Gender Shades*. FAT*.
Hardt et al. (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.
Kleinberg et al. (2017). *Inherent Trade-Offs in the Fair Determination of Risk Scores*. ITCS.
Madras et al. (2018). *Learning Adversarially Fair and Transferable Representations*. ICML.

**Skin tone e conditioning arquitetural**:
Schumann et al. (2023). *Consensus and Subjectivity of Skin Tone Annotation — MST-E*. CVPR.
Pereira et al. (2026). *Large-Scale Dataset and Benchmark for Skin Tone Classification in the Wild — SkinToneNet*. arXiv:2603.02475.
Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI.

**Refutação e diálogo crítico**:
Pangelinan et al. (2023). *Analyzing Bias in Race Classification*. FAccT.
AlDahoul et al. (2024). *FaceScanPaliGemma*. arXiv.

**Diversidade fenotípica intra-Latinx**:
Telles (2014). *Pigmentocracies: Ethnicity, Race, and Color in Latin America*. UNC Press.
Bryc et al. (2015). *The Genetic Ancestry of African Americans, Latinos, and European Americans across the United States*. AJHG.
Pew Research (2017). *Hispanic Identity Fades Across Generations*.

Bibliografia consolidada de 104 referências disponível em
`docs/tese/referencias.bib` do repositório do projeto.

## 9. Local de execução

Toda a pesquisa será conduzida **em ambiente computacional
próprio** do mestrando e em infraestrutura de computação
disponibilizada pela Unifesp / ICT ou por serviços de computação
em nuvem contratados individualmente (Google Colab Pro, AWS
educacional ou equivalente). Não há uso de laboratórios
institucionais que envolvam seres humanos ou animais.

## 10. Declaração final

Declaro, para os devidos fins, que o projeto de pesquisa aqui
descrito não envolve, direta ou indiretamente, seres humanos como
sujeitos de pesquisa, nem faz uso de animais vertebrados vivos,
enquadrando-se no Art. 8º da Resolução nº 200/2021 do Conselho
Universitário da Universidade Federal de São Paulo. Comprometo-me
a submeter novo protocolo ao Comitê de Ética em Pesquisa da Unifesp
caso, no curso da pesquisa, sobrevenha qualquer alteração de escopo
que envolva seres humanos ou animais.

---

São Paulo, 08 de julho de 2026.

Marcello Ozzetti — Mestrando
Prof. Dr. Marcos Quiles — Orientador / Pesquisador Responsável
