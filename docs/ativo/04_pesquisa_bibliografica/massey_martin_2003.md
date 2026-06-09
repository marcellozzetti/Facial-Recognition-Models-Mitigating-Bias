---
name: massey-martin-2003
status_verificacao: VERIFIED
autores: [Douglas S. Massey, Jennifer A. Martin]
ano: 2003
titulo: "The NIS Skin Color Scale"
venue: "Princeton University / NIS technical documentation"
tipo_publicacao: technical_report
arxiv_id: null
doi: null
url_primario: https://scales.arabpsychology.com/s/new-immigrant-survey-skin-color-scale/
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~10 (estimado)
lente_disrupcao: nenhuma (paper-instrumento)
fonte_leitura: WebSearch + summaries (NSF PAR, Survey Practice, ANES 2012, ResearchGate, Frontiers in Sociology 2024). Documento técnico do NIS sem PDF padrão publicamente baixável.
---

# The NIS Skin Color Scale (Massey & Martin, 2003)

> **Escala sociológica de 11 pontos** desenvolvida para o **New
> Immigrant Survey** (NIS) — primeiro instrumento padronizado para
> medir colorism em pesquisa empírica social americana.

## 1. Resumo do problema atacado

Em 2003, Douglas Massey (sociólogo, Princeton — figura central de
estudos de imigração) e Jennifer Martin precisavam de **instrumento
quantitativo de tom de pele** para o New Immigrant Survey, estudo
longitudinal de imigrantes nos EUA. Fitzpatrick (6 pontos) era
**dermatológica**, não adaptada a survey social. Felix von Luschan
(36) era **descontinuada** por excesso de granularidade e origem
eugenista.

Solução: escala de **11 pontos** com **imagens de mãos** (cuffs
visíveis), do mais claro (0=albinismo) ao mais escuro (10). Ancorada
em proxy visual ambíguo (mão estendida) — anotador internaliza
escala e anota sem mostrar ao respondente.

## 2. Método

- **11 pontos**, valores **0** (albinismo / mais claro possível) a **10**
  (mais escuro possível).
- **Material visual:** 10 fotografias de mãos estendendo-se de
  camisa branca, em gradiente de tonalidade.
- **Protocolo de anotação:** **entrevistadores memorizam a escala**;
  **respondente NUNCA vê** o chart. Mitiga viés de auto-classificação.
- **Treinamento de entrevistadores** com data collection managers
  pré-fieldwork; entrevistadores **não sabem onde eles próprios
  pontuam** na escala (evita anchoring effect).

## 3. Datasets e setup experimental

- **New Immigrant Survey (NIS)**: amostra nacional longitudinal de
  imigrantes adquirindo green card.
- Posteriormente adotada por:
  - **National Longitudinal Survey of Youth 1997 (NLSY97)**.
  - **General Social Survey (GSS)** — instrumento canônico em
    sociologia americana.
  - **Fragile Families and Child Wellbeing Study**.
  - **American National Election Studies (ANES)** 2012 time series.

## 4. Métricas reportadas

- **Inter-rater reliability** validado por Hersch (2008) — confirma
  confiabilidade da medida.
- **Correlação com outcomes sociais** (renda, educação, discriminação
  percebida) — central no programa de pesquisa de **colorism**.

## 5. Resultados principais

**Achados centrais do programa de pesquisa Massey-Martin sobre
colorism:**

- Imigrantes de pele mais escura (NIS scale ≥6) têm **renda
  significativamente menor** que imigrantes de pele clara (≤3),
  **controlando para raça/etnia auto-declarada**.
- Efeito persiste **dentro de cada grupo racial** — colorism existe
  intra-categoria racial, não só inter-categoria.
- Discriminação no mercado de trabalho americano é **modulada por
  tom de pele**, não apenas por categoria racial declarada.

**Implicação teórica:**

> **Tom de pele e categoria racial são variáveis distintas com
> efeitos sociais independentes**. Nenhuma é redutível à outra.

## 6. Limitações declaradas pelos autores

- Escala observada **por entrevistador**, não auto-declarada —
  pode introduzir viés do anotador (mitigado por treinamento).
- 11 pontos é **trade-off** entre granularidade e confiabilidade
  (precedente para MST 10-point discussão sobre cognitive load).
- Mãos como proxy podem não refletir **tom de pele facial**
  precisamente (variação rosto vs mãos por exposição UV).

## 7. Limitações que identifiquei (leitura crítica)

- **Não validada para fairness em ML** — instrumento sociológico,
  não biométrico.
- **Imagens originais** desatualizadas; ANES 2012 atualizou para
  versão digital.
- **Tom de pele de mãos ≠ rosto** — diferença observável,
  especialmente em populações com forte exposição solar
  ocupacional.
- **Escala observacional** (entrevistador anota) — depende de
  treinamento e produz variação inter-rater (Schumann 2023
  documenta padrão similar para MST).

## 8. Relação com nossa pesquisa

**Centralidade conceitual moderada:**

1. **Precedente histórico-metodológico para MST:** Monk Skin Tone
   (Schumann 2023) tem origem **sociológica** semelhante (Ellis Monk
   é sociólogo de Harvard, mesmo campo de Massey). Lineage:
   **Felix von Luschan → Massey-Martin 2003 → MST 2023**.
2. **Evidência empírica para "colorism vs racism" como variáveis
   independentes:** sustenta nossa frente Q10 (matriz skin tone ×
   race) — não estamos cruzando variáveis redundantes, são
   **constructos sociais distintos**.
3. **Protocolo de anotação treinada** com avaliação cega do entrevistador
   sobre sua própria posição na escala — **template metodológico**
   para Fase 2 do Q10 (validação manual MST).
4. **Citação contextual** para "tom de pele tem efeitos sociais
   independentemente de raça" — argumento ético/social da
   dissertação.

## 9. Pontos para citar / posicionar

- *"Massey e Martin (2003), em desenvolvimento metodológico para o
  New Immigrant Survey (NIS), introduziram a escala de tom de pele
  de 11 pontos hoje conhecida como NIS Skin Color Scale — instrumento
  amplamente adotado em estudos sociológicos americanos (GSS, NLSY97,
  Fragile Families, ANES) para documentar efeitos de colorism
  independentes de categoria racial."*
- *"Crucial à formulação da nossa frente Q10 é a observação
  empírica de Massey-Martin: tom de pele e categoria racial têm
  efeitos sociais distintos, sugerindo que a auditoria de fairness
  em sistemas faciais deve operar nas duas dimensões
  simultaneamente, não em uma como proxy da outra."*

## 10. Arquivos relacionados

- Frontiers in Sociology 2024 (Colorism in immigrant earnings):
  https://www.frontiersin.org/journals/sociology/articles/10.3389/fsoc.2024.1494236/full
- NSF PAR (Measuring Skin Color survey paper): par.nsf.gov/servlets/purl/10352585
- ANES 2012 documentação: electionstudies.org/2012-time-series-study-skin-color
- Entradas relacionadas: [[schumann_2023]] (MST — sucessor),
  [[fitzpatrick_1988]] (escala dermatológica alternativa),
  [[fuentes_2019]] (race ≠ biology fundamenta colorism
  ser variável independente).
- **Responde Q14 em [[_perguntas]]** (escalas alternativas).

## 11. Trabalhos sugeridos pelos autores (Future Work)

Programa de pesquisa Massey-Martin sobre colorism continuou e
continua:

- **Aplicar escala a outros contextos nacionais** — feito (Brasil,
  México, etc.). ❌ Fora do escopo.
- **Investigar mecanismos do colorism** (discriminação direta vs
  estereótipos vs autodescriminação). ❌ Fora do escopo computacional.
- **Combinar com indicadores genéticos / fenotípicos diretos** —
  alguns estudos posteriores fizeram, mas raros. ⚠ Tangencial.
- **Atualizar escala para meios digitais** — feito por ANES 2012.
  ❌ Fora do escopo.
- **Validação cross-cultural da escala** — feito por Hersch (2008)
  e outros. ❌ Já feito.

**Direção não-explorada (potencial nossa contribuição):**

- **Cruzar escala Massey-Martin (ou MST) com classificadores
  faciais** modernos — Q10 ✅ **GAP CENTRAL**.

## 12. Análise crítica do método

### (a) Rigor formal

- **Escala de 11 pontos (0-10) com imagens de mãos** — instrumento
  metodologicamente claro.
- **Protocolo de anotação** sofisticado: entrevistadores memorizam,
  respondente não vê, treinamento prévio, anonimato do próprio
  ponto da escala.
- **Inter-rater reliability validada** por Hersch (2008).
- **Correlação com outcomes sociais** (colorism) — programa de
  pesquisa empiricamente robusto.

### (b) Reprodutibilidade

- ⚠ **Documento técnico do NIS**, não paper acadêmico tradicional —
  ficha construída via WebSearch + múltiplas fontes secundárias
  (NSF PAR, Survey Practice, ANES 2012, ResearchGate).
- ✅ Escala adotada em múltiplos estudos longitudinais (NIS, NLSY97,
  GSS, Fragile Families, ANES) — robustez por replicação ampla.
- ⚠ Imagens originais não publicamente baixáveis em alta resolução.

### (c) Aplicabilidade ao pipeline v3.2

- **Não é técnica computacional** — é precedente histórico-
  metodológico para MST.
- **Protocolo de anotação treinada** com avaliação cega é template
  para Fase 2 do Q10 (validação manual MST em FairFace).
- **Evidência empírica colorism vs racism** sustenta nossa frente
  Q10 (skin tone × race como variáveis independentes).

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| 11 pontos (0-10) | ✅ Justificada — trade-off granularidade × IAA |
| Mãos como proxy de tom | ⚠ Limitação — rosto pode diferir |
| Entrevistador anota (não autoreporte) | ✅ Justificada — mitiga viés de auto-classificação |
| Respondente não vê escala | ✅ Justificada — evita priming |
| Treinamento prévio + cegueira do próprio ponto | ✅ Justificada — minimiza anchoring effect |
| Sem validação para ML/biometria | ✅ Choice — escopo sociológico deliberado |

### (e) Conexão com R5/R6

- [[fitzpatrick_1988]]: alternativa dermatológica de 6 pontos. NIS é
  sociológica de 11.
- [[schumann_2023]] MST: descendente moderno. Lineage Felix von
  Luschan (36) → Massey-Martin 2003 (11) → MST 2023 (10).
- [[buolamwini_2018]] Gender Shades: adopter de Fitzpatrick, não NIS.
- [[fuentes_2019]]: AAPA statement sustenta colorism como variável
  independente de raça.
- **Implicação para v3.2**: Massey-Martin é **precedente histórico**
  para MST. Cita-se como referência da linhagem; instrumento
  operacional na dissertação é MST.
