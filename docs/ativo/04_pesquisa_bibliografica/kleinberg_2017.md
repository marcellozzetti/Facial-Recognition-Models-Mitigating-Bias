---
name: kleinberg-2017
status_verificacao: VERIFIED
autores: [Jon Kleinberg, Sendhil Mullainathan, Manish Raghavan]
ano: 2017
titulo: "Inherent Trade-Offs in the Fair Determination of Risk Scores"
venue: "8th Innovations in Theoretical Computer Science Conference (ITCS 2017), LIPIcs vol 67"
tipo_publicacao: conference
arxiv_id: "1609.05807"
doi: "10.4230/LIPIcs.ITCS.2017.43"
url_primario: https://arxiv.org/abs/1609.05807
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-04
n_referencias_paper: ~20
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/kleinberg_2017_impossibility.pdf)
---

# Inherent Trade-Offs in the Fair Determination of Risk Scores (Kleinberg, Mullainathan & Raghavan, 2017)

> **Teorema da impossibilidade da fairness** — prova matematicamente
> que três critérios razoáveis de fairness **não podem ser
> simultaneamente satisfeitos** exceto em casos altamente
> restritos. Citado em [[survey_mehrabi_2021]] como evidência de que
> escolha de métrica é **decisão de design**, não escolha "neutra".

## 1. Resumo do problema atacado

Após COMPAS controvérsia (algoritmo de risco de reincidência
acusado de viés racial), comunidade ML/CS debate qual definição de
fairness aplicar a risk scores. Kleinberg et al. demonstram **teorema
formal**: três condições simultâneas geralmente impossíveis.

## 2. Método

### 2.1 As três condições de fairness

Para risk score que estima P(Y=1) condicionado a features:

1. **Calibration within groups**: para todo score s, P(Y=1 | score=s,
   A=0) = P(Y=1 | score=s, A=1).
2. **Balance for the positive class**: E[score | Y=1, A=0] = E[score | Y=1, A=1].
3. **Balance for the negative class**: E[score | Y=0, A=0] = E[score | Y=0, A=1].

### 2.2 Teorema central

**Se P(Y=1 | A=0) ≠ P(Y=1 | A=1) (taxas-base diferentes entre
grupos), então não existe risk score que satisfaça as 3 condições
simultaneamente — exceto em casos onde score é trivialmente perfeito
ou trivialmente uniforme.**

### 2.3 Implicação

**Trade-off inevitável**: ao escolher 2 das 3 condições para
satisfazer, sacrifica-se a terceira. **A escolha é decisão de
policy/ética**, não decisão neutra de engenharia.

## 3. Datasets e setup experimental

Não há experimento empírico — paper é **teórico**. Discussão usa
COMPAS como motivação narrativa.

## 4. Métricas reportadas

N/A — paper de prova matemática.

## 5. Resultados principais

**Teorema 1.1 (paraphrased)**: dado que taxas-base de Y diferem
entre grupos, e score é não-trivial, calibration + balance-positive
+ balance-negative são incompatíveis.

**Corolário operacional**: COMPAS é calibrated entre raças (ProPublica
contestou isso) MAS não é balance-negative (afro-americanos com
baixo risco real recebem scores médios maiores que brancos com baixo
risco real). Esta tensão **não é falha de implementação** — é
matemática.

## 6. Limitações declaradas pelos autores

- Resultado é **assintótico** — em datasets finitos pode haver
  aproximações.
- Não cobre **definições baseadas em representação** (Zemel 2013,
  Madras 2018).
- Não trata **causal fairness** — paralelo de Pearl/Kusner.

## 7. Limitações que identifiquei

- Resultado é sobre **risk scores** (output contínuo). Para
  **classification** (output discreto), o teorema se aplica via
  derivação mas formulação direta exige adaptação.
- **Caso binário** (Y, A binários). Para race 7-class, generalização é
  via one-vs-rest ou worst-case.
- **Pressupõe taxas-base diferentes**. Em FairFace 7-class, taxa-base
  P(Y=k) varia por classe (White 19%, Middle East 11%); então as
  condições do teorema **se aplicam**.

## 8. Relação com nossa pesquisa

**Citação obrigatória** para justificar:

1. **Por que reportamos múltiplas métricas** (Q05 — triangulação DR
   + worst-class + CV): Kleinberg prova que **uma métrica única não
   captura fairness**. Triangulação não é redundância — é resposta
   ao teorema.
2. **Por que escolha de métrica é decisão de design**: nossa
   pesquisa **opta por DR** (max/min ratio) por interpretabilidade
   operacional; Kleinberg sustenta que outras escolhas são igualmente
   defensáveis para outras prioridades.
3. **Honestidade na limitação §6**: tese v3.1 reconhece que não
   podemos satisfazer "todas as fairness simultaneamente" — citação
   formal a Kleinberg ancora essa honestidade.

**Para v3.2**: ao avaliar antes-vs-depois (sugestão do orientador),
**não há "melhora absoluta"**. Há trade-offs entre métricas.
Reportar honestamente as 3 (DR, worst-class F1, CV) + interpretar
qual prioridade cada uma reflete é a postura correta.

## 9. Pontos para citar

- *"Kleinberg, Mullainathan e Raghavan (ITCS 2017) demonstram
  matematicamente que três critérios razoáveis de fairness —
  calibration within groups, balance for positive class, balance for
  negative class — não podem ser simultaneamente satisfeitos quando
  taxas-base do target diferem entre grupos, exceto em casos
  trivialmente restritos."*
- *"A escolha de métrica de fairness adotada nesta dissertação —
  triangulação de Disparity Ratio, worst-class F1 e Coefficient of
  Variation — não pretende capturar 'fairness universal', mas
  reflete a posição teoricamente sustentada de Kleinberg et al.
  (2017) de que a escolha entre noções de fairness é decisão de
  prioridade ética, não escolha tecnicamente neutra."*

## 10. Arquivos relacionados

- PDF: `pdfs/kleinberg_2017_impossibility.pdf` (gitignored).
- DOI: 10.4230/LIPIcs.ITCS.2017.43 (LIPIcs open access).
- Entradas relacionadas: [[survey_mehrabi_2021]] (cita teorema na
  §4 das definições), [[hardt_2016]] (paper paralelo, mesma época,
  mesma motivação COMPAS), [[fuentes_2019]] (fundamento ético da
  posição "race é construto" complementa Kleinberg matemático).

## 11. Trabalhos sugeridos pelos autores (Future Work)

- **Investigar definições alternativas** que evitam impossibilidade.
  ⚠ Direção teórica.
- **Estudar trade-offs em casos com taxas-base similares** — onde
  teorema não se aplica. ❌ Não é nosso caso.
- **Aplicar a outras domínios** além de risk scores. ✅ Adoção ampla
  pela comunidade em classification, recommendation, ranking.
- **Causal fairness** como possível alternativa que escapa
  impossibilidade. ⚠ Direção paralela (Pearl, Kusner).

## 12. Análise crítica do método

### (a) Rigor formal

- **Paper teórico puro**: Teorema 1.1 com prova matemática formal.
- **Três condições de fairness** matematicamente claras (calibration,
  balance positive, balance negative).
- **Impossibilidade matemática** demonstrada exceto em casos
  trivialmente restritos.
- **Limitação reconhecida**: resultado é assintótico — em datasets
  finitos pode haver aproximações.

### (b) Reprodutibilidade

- ✅ Paper teórico — provas verificáveis matematicamente.
- ✅ Conferência ITCS (theoretical CS top venue), LIPIcs open access.
- ⚠ COMPAS discussão é narrativa — não experimental.
- ✅ Adoção ampla pela comunidade de fair ML como princípio
  fundamental.

### (c) Aplicabilidade ao pipeline v3.2

- **Justifica triangulação de métricas**: nossa pesquisa reporta DR +
  worst-class F1 + CV simultaneamente — instância prática do
  princípio de Kleinberg.
- **Limitação para race 7-class**: teorema é binário; generalização
  para multi-classe via one-vs-rest ou worst-case.
- **Calibra expectativas**: não há "fairness universal" — escolha de
  métrica é decisão de design.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Três condições escolhidas | ✅ Justificada — representativas da literatura |
| Risk scores (output contínuo) | ⚠ Choice — generalização para classificação via derivação |
| Y, A binários | ⚠ Limitação reconhecida |
| Assume taxas-base diferentes | ⚠ Pressuposto — caso comum |
| Discussão COMPAS narrativa | ⚠ Choice — paper teórico |
| Sem proposta de solução prática | ✅ Choice — escopo teórico deliberado |

### (e) Conexão com R5/R6

- [[hardt_2016]]: paper paralelo, mesma época, mesma motivação
  (COMPAS). Hardt propõe definição (EO_h); Kleinberg prova
  impossibilidade de combinar definições.
- [[zemel_2013]]: Kleinberg restringe quantas fairness Zemel pode
  satisfazer.
- [[madras_2018]] LAFTR: a tradução de objetivo adversarial → noção
  de fairness de Madras opera sob a sombra de Kleinberg.
- [[dehdashtian_2024]] U-FaTE: quantifica empiricamente o trade-off
  que Kleinberg prova matematicamente.
- [[mehrabi_2021]] survey: cita o teorema na Seção 4 das
  definições.
- [[fuentes_2019]]: combinação Kleinberg matemática + Fuentes
  sociológica = dupla incompatibilidade conceitual da fairness sob
  taxonomia racial.
- **Implicação para v3.2**: Kleinberg é **citação obrigatória** na
  discussão sobre escolha de métrica. Nossa triangulação (DR +
  worst-class + CV) é resposta ao teorema, não tentativa de evadi-lo.
