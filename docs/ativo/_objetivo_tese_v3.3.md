# Objetivo da tese — v3.4 (pós-revisão bibliográfica completa)

> **Versão**: 3.4 — incorpora recomendações do cross-reference
> sistemático tese × 101 fichas
> (`_validacao_cross_reference_v3.md`).
> **Versão anterior**: v3.3 (2026-06-10, pós-aprovação do pipeline
> pelo orientador na reunião de 08/jun).
> **Atualizada**: 2026-06-15.
> **Propósito**: âncora narrativa para escrita da qualificação
> (deadline 15-jul-2026 — entrega da primeira revisão ao orientador).
>
> **Mudanças vs v3.3**:
> 1. OE-6 promovido (era H6) — decomposição de variância pixel info × skin tone como objetivo específico formal.
> 2. Citação central de Wang+Deng BUPT 2022 como precedente metodológico (já argumentava skin tone > race labels).
> 3. Adicionadas referências brasileiras/portuguesas como diferencial.
> 4. Coeficiente de Gini (Yang IJCV 2022) considerado como métrica auxiliar.
> 5. Estudo comparativo Cap 2 formalizado em 4 configurações (A baseline / B FiLM-MST / C Gated FiLM / D FiLM-CLIP).

## 1. Storytelling aprovado pelo orientador

A tese deve seguir a estrutura narrativa (confirmada na reunião):

1. **Contexto**
2. **Problemas existentes**
3. **O que tem sido feito** (estudos recentes)
4. **O que falta ser explorado** (gap)
5. **Objetivo**
6. **Como será feito** (metodologia)

## 2. Objetivo geral (1 parágrafo)

> **Esta dissertação tem como objetivo desenvolver e avaliar um
> pipeline de classificação racial em imagens faciais que incorpora
> tom de pele (escala Monk Skin Tone) como sinal auxiliar condicionante
> via mecanismo arquitetural, com o propósito de mitigar viés racial
> demonstrável no estado-da-arte atual — particularmente a
> disparidade severa entre classes raciais bem representadas
> (Black F1 ≈ 90%) e classes sub-representadas (Latinx F1 ≈ 60%)
> documentada em modelos como FaceScanPaliGemma sobre o dataset
> FairFace. A contribuição principal é a primeira instância
> empírica documentada do uso de tom de pele explícito como contexto
> arquitetural para race classification multi-classe, com avaliação
> rigorosa via triangulação de métricas (DR + worst-class F1 + F1
> macro, complementadas por EO_h e EOD por classe em ablation
> intersectional race × gender), e demonstração de fair transferência
> do mecanismo para downstream face recognition.**

## 3. Objetivos específicos (5)

### Objetivo específico 1 — Auditoria fenotípica do FairFace (Cap 1)

Quantificar a distribuição cruzada **MST × race classes** sobre o
FairFace via SkinToneNet (Pereira 2026) com validação humana em
subset. **Entregar a primeira matriz pública dessa distribuição**.

- **Hipótese testada**: H3 — Latinx tem spread MST amplo (≥ 5 das
  10 classes MST).
- **Métricas**: histograma MST por raça, % overlap entre raças,
  validação manual via Prolific (~700 imgs × 3 anotadores).

### Objetivo específico 2 — Avaliar modelos pré-treinados de skin tone (Cap 1)

> **Recomendação do orientador (2026-06-08)**.

Conduzir avaliação metodológica de modelos pré-treinados disponíveis
para classificação MST (SkinToneNet, Casual Conversations baseline,
Google API, alternativas HuggingFace) e **justificar a escolha do
modelo adotado** com critérios de desempenho, generalização e
disponibilidade.

- **Saída**: tabela comparativa + decisão fundamentada + protocolo
  de validação.

### Objetivo específico 3 — Pipeline condicionado para race classification (Cap 2)

Implementar e avaliar pipeline **SkinToneNet → ConvNeXt-T + FiLM →
race classifier** sobre FairFace, comparando contra 6 baselines
(ConvNeXt-T puro, ResNet-34, FSCL+, Group DRO, FineFACE, Adversarial
debiasing).

**Adicional** (pós-reunião): comparar **FiLM-conditioning vs
CLIP-conditioning** em ablation arquitetural — endereçar
recomendação do orientador de testar mecanismo moderno alternativo.

- **Hipóteses testadas**: H1 (pipeline funciona), H2 (ConvNeXt-T
  puro vs ResNet-34), H4 (erros Latinx em zonas overlap MST).
- **Métricas**: F1 macro, DR, worst-class F1, EO_h por classe e EOD
  por classe (ablation race × gender).

### Objetivo específico 4 — Fair transferência para face recognition (Cap 3)

Demonstrar empíricamente que o backbone fair-treinado no Cap 2
transfere a propriedade fair para tarefa downstream de **face
recognition** sobre RFW ou BFW, com controle explícito de **pixel
information** como confounder (resposta a Pangelinan 2023).

- **Hipóteses testadas** (revisadas pós-Pangelinan):
  - **H5** (cautelosa): condicionamento MST melhora fairness em FR
    *quando pixel info é controlada*.
  - **H6** (nova): disparity residual em Black/African é
    predominantemente explicada por pixel info, NÃO por skin tone
    (decomposição de variância).
- **Métricas**: TAR @ FAR fixo por raça, ε-DEO transferido.

### Objetivo específico 5 — Síntese metodológica (Cap 4)

Decompor o erro de classificação de Latinx em **componente
fenotípico (irredutível pelo overlap MST)** vs **componente
algorítmico (mitigável)** via análise cruzada dos resultados dos
Caps 1-3. **Quantificar quanto da disparidade é estrutural** (fronteira
de classificação ambígua por sobreposição MST) **vs quanto é
mitigável** (resíduo pós-conditioning).

- **Métrica**: % variance accounted by MST overlap vs % attributable
  to model bias.
- **Resultado esperado**: decomposição que sustenta a discussão
  ética/social sobre limites estruturais de race classification
  multi-classe.

### Objetivo específico 6 — Decomposição quantitativa pixel info × skin tone (Cap 4) — **NOVO v3.4**

> **Promoção da hipótese H6 a objetivo específico formal** após
> leitura integral de Pangelinan et al. (2023, Camada 1 VERIFIED).

**Quantificar empíricamente a contribuição relativa de duas
variáveis** ao gap de disparidade entre raças:

1. **Pixel information** (face-pixel fraction via segmentation,
   replicando metodologia BiSeNet de Pangelinan 2023).
2. **Skin tone** (vetor MST 10-dim via SkinToneNet).

**Hipótese H6 formalizada**: disparity residual no Black/African após
o conditioning é predominantemente explicada por pixel info,
**não** por skin tone — esperando ≥ 70 % da variância
do gap atribuível a face-pixel fraction.

- **Métrica**: R² na decomposição de variância (linear regression
  do gap residual sobre pixel info + MST).
- **Auxiliar**: coeficiente de Gini (Yang et al. IJCV 2022, BUPT)
  para reportar long-tailedness do FairFace 7-class.
- **Resultado esperado**: confirmação ou refutação quantitativa da
  refutação central de Pangelinan, transformando o ponto de tensão
  metodológica em **contribuição original quantitativa** (C6
  reformulada).

### Status do OE-6

- **Decisão pendente do orientador** (item agendado para próxima reunião).
- **Provisoriamente incluído** porque (a) já está endossado pela leitura integral de Pangelinan, (b) reforça defensividade da banca, (c) pode ser rebaixado para H6 (sem OE) se orientador preferir.

## 4. Contribuições científicas (revisadas v3.4)

| ID | Contribuição | Originalidade |
|---|---|---|
| **C1** | Avaliação metodológica de modelos pré-treinados MST + protocolo de escolha | Recomendação orientador 2026-06-08 |
| **C2** | Matriz pública MST × race do FairFace | Não publicado: Pereira 2026 só agrega |
| **C3** | Primeira aplicação de FiLM-conditioning a fairness facial | Sem precedente direto (Perez 2018 §11 declara fairness como Future Work) |
| **C4** | Triangulação de métricas multi-classe (DR + worst-class F1 + EO_h/EOD por classe) | Baseado em Hardt 2016, original em race 7-class. Coeficiente de Gini (Yang IJCV 2022) como métrica auxiliar de long-tailedness. |
| **C5** | Demonstração empírica de fair transferência classification → FR | LAFTR é teórico; Aguirre 2023 é NLP; CV é nova |
| **C6** | Decomposição de variância **fenotípico × algorítmico** com quantificação explícita pixel info × skin tone | Diagnóstico inédito. **Endossada por Pangelinan 2023** (refutação central — Camada 1 VERIFIED). Reforçada via OE-6. |
| **C7** | Estudo comparativo de mecanismos de conditioning em race fairness — **4 configurações**: A baseline / B FiLM-MST (proposta) / C Gated FiLM (ablação) / D FiLM-CLIP (avaliação alternativa) | Recomendação orientador 2026-06-15 + decisão arquitetural pós-avaliação de 8 candidatos (`_decisao_arquitetural_film.md`) |

## 5. Hipóteses revisadas v3.3 (6 hipóteses, era 5)

| ID | Hipótese | Critério de confirmação | Critério de refutação |
|---|---|---|---|
| **H1** | Pipeline MST + FiLM → ConvNeXt-T supera baseline ResNet-34 em F1 macro ≥+2pp E reduz DR ≥20% | Ambos satisfeitos | Qualquer um abaixo |
| **H2** | ConvNeXt-T puro ganha +2 a +5pp sobre ResNet-34; Latinx F1 ≈60% (±3pp invariante) | Ganho no range E Latinx invariante | Fora do range OU Latinx muda |
| **H3** | Matriz MST × race mostra Latinx com spread ≥ 5 das 10 classes MST | Spread ≥ 5 | Spread < 5 |
| **H4** | ≥ 50% dos erros Latinx no baseline estão em zonas MST de overlap | %_overlap ≥ 50% | %_overlap < 50% |
| **H5** (revisada) | Condicionamento MST mantém ou melhora fairness em FR **com quality control de pixel info** | ≥+3pp em Black/African após normalização pixel info | < +3pp ou degradação |
| **H6** (NOVA) | Disparity residual após conditioning é explicada predominantemente por pixel info (Pangelinan 2023) | ≥ 70% variance explicada por pixel info | < 70% (sugere bias algorítmico residual) |

## 6. Foco da tese (do orientador)

**Melhorar a acurácia dos modelos de reconhecimento de imagem para
mitigar vieses, com foco em viés racial / cor de faces.**

- **Tarefa central**: race classification 7-class no FairFace.
- **Mecanismo central**: tom de pele MST como sinal auxiliar.
- **Resultado esperado**: redução documentada de disparity entre
  raças.
- **Aplicação downstream**: face recognition.

## 7. Storytelling — esqueleto para escrita

### Contexto

Sistemas de reconhecimento facial estão em uso massivo (catraca,
banco, fronteira, identificação policial). A literatura documenta há
quase uma década (Buolamwini & Gebru 2018) que esses sistemas
**falham desproporcionalmente em grupos sub-representados**.

### Problemas existentes

- **Estado-da-arte tem disparity severa**: FaceScanPaliGemma
  (AlDahoul 2026) atinge F1 macro 75% mas F1 Latinx 60%, F1 Black
  90% — gap de 30pp entre as raças.
- **Balanceamento de dados não basta**: FairFace é balanceado por
  design, mas a disparidade persiste (Karkkainen 2021, Kolla 2022).
- **Mitigação algorítmica atual** (FSCL, Group DRO, Adversarial)
  não foi sistematicamente testada em race classification multi-classe.

### O que tem sido feito (estudos recentes)

- **Datasets balanceados**: FairFace (Kärkkäinen & Joo 2021),
  RFW (Wang et al. 2019), BFW (Robinson et al. 2020),
  BUPT-Balancedface (Wang, Zhang & Deng — BUPT 2022).
- **Skin tone como dimensão alternativa**: Schumann et al. 2023
  (Monk Skin Tone), Pereira Matias et al. 2026 (SkinToneNet),
  Porgali et al. 2023 (CCv2 — Meta), Wang & Deng 2022
  (**BUPT — já argumentavam em 2022 que skin tone é label mais
  preciso/científico que race**, antecipando a decisão metodológica
  desta dissertação).
- **Mitigação algorítmica**: Park et al. 2022 (FSCL+), Sagawa et al.
  2020 (Group DRO), Manzoor & Rattani 2024 (FineFACE), Liu et al.
  2025 (BNMR — FAccT).
- **Vision-language models**: AlDahoul et al. 2024 (FaceScanPaliGemma
  — Nature SR 2026), Luo et al. 2024 (FairCLIP — CVPR 2024),
  Dehdashtian et al. 2024 (FairerCLIP — ICLR 2024).
- **Produção científica brasileira**: Parraga et al. (PUCRS MALTA Lab,
  ACM CSur 2025) — survey de fairness em deep learning para
  vision e language, financiado por Motorola Mobility Brasil + CAPES.
- **Produção científica portuguesa**: grupo Univ. Porto + INESC TEC
  (Pedro C. Neto, Ana F. Sequeira, Rafael Mamede et al.) com 4
  trabalhos no corpus (MST-KD 2024, Massively Annotated 2024,
  Occlusion Bias 2024) e Univ. Coimbra ISR (VOIDFace 2025).

### O que falta ser explorado (gap)

- **Skin tone como sinal arquitetural condicionante** em race
  classification multi-classe — sem precedente direto. **Perez et
  al. (2018, FiLM) declara explicitamente em §11 (Future Work) que
  aplicação a fairness não foi testada**, configurando lacuna
  formal.
- **Matriz pública MST × race** para FairFace — Pereira 2026 audita
  oito datasets via SkinToneNet mas **não publica a cross-tabulation
  MST × race**.
- **Decomposição quantitativa pixel info × skin tone** como
  contribuições explicativas independentes do gap residual após
  conditioning — endereçando refutação central de Pangelinan et al.
  (2023, Camada 1 VERIFIED) que documenta pixel info como
  confounder primário.
- **Fair transferência empírica em face recognition CV** — Madras
  et al. (2018, LAFTR) é puramente teórico e Aguirre & Dredze (2023)
  demonstra empiricamente apenas em NLP.

### Objetivo

Ver Seção 2 acima.

### Como será feito

Pipeline em 6 etapas (aprovado pelo orientador). Cap 1 (MST audit) +
Cap 2 (race + conditioning) + Cap 3 (fair transfer to FR) + Cap 4
(síntese decompositiva).

Triangulação de métricas (Hardt 2016 base, multi-classe nossa).

Comparação contra 6 baselines + ablation FiLM vs CLIP-conditioning.

## 8. Próximas ações imediatas

1. ✅ Registrar reunião (2026-06-08 doc criado).
2. ✅ Criar 7 fichas R6 (Aguirre VERIFIED + 6 OVERVIEW_ONLY).
3. ✅ Planejar Rodada 7 (este documento + _rodada_07_planejamento).
4. ✅ Atualizar objetivo e hipóteses para v3.3.
5. **PENDENTE**: executar buscas Rodada 7 e triar candidatos.
6. **PENDENTE**: setup Overleaf com template Unifesp/ICT.
7. **PENDENTE**: começar escrita da Introdução (Cap 1).
