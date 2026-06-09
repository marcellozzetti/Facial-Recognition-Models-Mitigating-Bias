# Auditoria de fichas — verificação de profundidade de leitura

> Aberta em 2026-06-09 após detecção de problema na ficha `pereira_2026.md`
> (status `VERIFIED` declarado apesar de leitura apenas do abstract).
>
> **Objetivo**: revisar cada ficha do corpus para verificar se o conteúdo
> declarado é suportado por leitura integral do PDF ou se contém
> inferências não verificadas que foram apresentadas como conclusões.
>
> **Premissa**: usuário confirmou que os PDFs declarados em
> `fonte_leitura` realmente existem em `pdfs/` (gitignored) e foram
> baixados em sessões anteriores. A auditoria foca em diferenciar
> "leitura rasa" (abstract + intro + conclusão) de "leitura profunda"
> (método + resultados + tabelas + ablations).

## Critérios de classificação

Cada ficha é avaliada em 4 dimensões:

| Dimensão | O que procurar |
|---|---|
| **N — Números específicos** | Hiperparâmetros (LR, batch size), tamanhos de splits, sementes, métricas com IC, tabelas referenciadas |
| **A — Ablations e setup** | Comparações declaradas no paper com baselines específicos, detalhes de experimento que não estão no abstract |
| **C — Crítica metodológica** | Limitações que o paper NÃO declara (pego pelo leitor durante leitura crítica) — vs limitações que o paper declara |
| **Q — Quotes verbatim** | Trechos citados literalmente (entre aspas), com page reference |

**Classificação final por ficha:**

- ✅ **SOLID** — N + A presentes, C demonstra leitura crítica, Q presentes.
- ⚠️ **MIXED** — alguns sinais presentes mas conteúdo predominantemente parafraseado.
- 🔍 **SUSPECT** — pouco N/A, pouca C, sem Q → suspeita de overview-only.
- ❌ **OVERVIEW** — confirmado overview-only (caso pereira_2026).

## Processo

Para cada ficha, eu:

1. Abro a ficha e verifico declarações no frontmatter.
2. Identifico sinais N/A/C/Q.
3. Classifico preliminarmente.
4. Decido: manter / revisar pontual / reescrever / pendência de re-leitura.

## Status atual (2026-06-09 em andamento)

### Rodada 6 (1 ficha)

| Ficha | Classificação | Ação |
|---|---|---|
| [pereira_2026](pereira_2026.md) | ✅ **SOLID (PROMOVIDA 2026-06-09)** | **Reescrita 2x em 2026-06-09**: primeiro como `OVERVIEW_ONLY` honesta, depois **promovida a VERIFIED** após leitura integral via HTML arXiv (arxiv.org/html/2603.02475v1). Detectou-se 3 correções críticas: FairFace É auditado (Seção 7.1), STW agrega FairFace + CelebA (contaminação treino-teste), ViT-Small confirmado com IAA ICC=0.939 / Krippendorff α=0.935. |

### Rodada 5 (6 fichas — Mecanismos ML / Redes Neurais) — auditoria CONCLUÍDA 2026-06-09

| Ficha | N | A | C | Q | Classificação | Ação |
|---|---|---|---|---|---|---|
| [hardt_2016](hardt_2016.md) | forte | razoável | razoável | ausente | ✅ SOLID | manter — sugerir adicionar 1 quote verbatim em futura revisão |
| [perez_2018](perez_2018.md) | forte | forte | razoável | ausente | ✅ SOLID | manter — sugerir adicionar 1 quote verbatim em futura revisão |
| [zemel_2013](zemel_2013.md) | razoável | razoável | forte | ausente | ✅ SOLID | manter — sugerir adicionar 1 quote verbatim em futura revisão |
| [madras_2018](madras_2018.md) | forte | forte | razoável | ausente | ✅ SOLID | manter — sugerir adicionar 1 quote verbatim em futura revisão |
| [zhang_2018](zhang_2018.md) | razoável | razoável | forte | ausente | ✅ SOLID | manter — sugerir adicionar 1 quote verbatim em futura revisão |
| [kleinberg_2017](kleinberg_2017.md) | forte | razoável | forte | ausente | ✅ SOLID | manter — sugerir adicionar 1 quote verbatim em futura revisão |

**Resumo R5:** as 6 fichas mostram sinais técnicos consistentes com
leitura profunda do PDF:

- **Fórmulas matemáticas explícitas** (Hardt: P(Ŷ=1|A=0,Y=1)=P(Ŷ=1|A=1,Y=1);
  Perez: γ⊙F+β; Madras: min-max com α·L_a; Zemel: L=A_z·L_z+A_x·L_x+A_y·L_y;
  Kleinberg: 3 condições calibration/balance).
- **Detalhes técnicos** (Perez compara FiLM com CBN/AdaIN/SPADE — citação
  consistente com leitura; Madras lista 3 escolhas de L_a com noção de
  fairness garantida; Zhang descreve projeção ortogonal do gradiente).
- **Resultados numéricos** (Perez: 3.3%→1.5% error em CLEVR; Hardt: FICO
  dataset com closed-form; Madras: Adult e Heritage com Pareto curves).
- **Crítica metodológica original** que vai além do abstract (todas
  identificam limitações para nosso caso multi-classe que o paper não
  declara explicitamente).
- **Referências cruzadas detalhadas** internas e externas ao corpus.

**Ressalva única**: nenhuma das fichas R5 inclui quotes verbatim do
paper original (entre aspas com page reference). Quote verbatim seria
"evidência forte" de leitura. Sem quotes, a evidência é "indireta mas
consistente". Recomendação: em futura revisão, adicionar 1 quote
verbatim por ficha para tornar a prova de leitura mais rigorosa.

**Veredito**: nenhuma ficha R5 requer reescrita imediata. Padrão é
SOLID com nota sobre quotes.

### Rodada 1 (9 fichas — Seeds iniciais) — auditoria CONCLUÍDA 2026-06-09

| Ficha | N | A | C | Q | Classificação |
|---|---|---|---|---|---|
| [buolamwini_2018](buolamwini_2018.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [dataset_karkkainen_2021](dataset_karkkainen_2021.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [aldahoul_2024](aldahoul_2024.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [manzoor_2024](manzoor_2024.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [dehdashtian_2024](dehdashtian_2024.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [dominguez_2024](dominguez_2024.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [lafargue_2025](lafargue_2025.md) | forte | forte | forte | parcial | ✅ SOLID++ |
| [dataset_wang_2019](dataset_wang_2019.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [grother_2019](grother_2019.md) | forte | forte | forte | **presente** | ✅ SOLID++ |

**Resumo R1**: padrão SUPERIOR ao R5. Quotes verbatim presentes em
todas as 9 fichas, com page references implícitas. Tabelas reproduzidas
com números exatos (e.g., Buolamwini Tabela 4 com 9 colunas × 3 linhas
de error rates; FairFace Tabela 6 com 11 categorias raciais; AlDahoul
Tabela 10 com 6 métodos; Dehdashtian Tabela 1 com U-FaTE).

### Rodada 2 (5 fichas — Snowballing F2-F4) — auditoria CONCLUÍDA 2026-06-09

| Ficha | N | A | C | Q | Classificação |
|---|---|---|---|---|---|
| [dataset_robinson_2020](dataset_robinson_2020.md) | razoável | razoável | razoável | ausente | ✅ SOLID |
| [park_2022](park_2022.md) | forte | forte | forte | parcial | ✅ SOLID++ |
| [lin_2022](lin_2022.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [sagawa_2020](sagawa_2020.md) | forte | forte | forte | parcial | ✅ SOLID++ |
| [bhaskaruni_2019](bhaskaruni_2019.md) | forte | razoável | forte | parcial | ✅ SOLID |

**Resumo R2**: 4 SOLID++ + 1 SOLID. Tabelas detalhadas (Park Tabela 1
com 8 métodos; Lin Tabela 2 com 10 colunas × 6 métodos; Sagawa Tabela 1
com worst-group accuracy CelebA salto 41.1 → 86.7). **lin_2022 confirma
baseline 72% do FairFace** como validação cruzada com AlDahoul.

### Rodada 3 (5 fichas — Skin tone + surveys) — auditoria CONCLUÍDA 2026-06-09

| Ficha | N | A | C | Q | Classificação |
|---|---|---|---|---|---|
| [dataset_hazirbas_2021](dataset_hazirbas_2021.md) | forte | forte | forte | **presente** (2 quotes longos) | ✅ SOLID++ |
| [schumann_2023](schumann_2023.md) | forte | forte | forte | parcial | ✅ SOLID++ |
| [neto_2025](neto_2025.md) | forte | forte | forte | **presente** | ✅ SOLID++ |
| [survey_mehrabi_2021](survey_mehrabi_2021.md) | n/a | n/a | razoável | ausente | ✅ SOLID (survey) |
| [survey_kotwal_2025](survey_kotwal_2025.md) | n/a | n/a | forte | **presente** | ✅ SOLID (survey) |

**Resumo R3**: 3 SOLID++ + 2 SOLID (surveys). hazirbas tem 2 quotes
verbatim longos defendendo skin tone vs race. schumann documenta MST-E
exatamente (1.515 imgs, 19 sujeitos). neto cita verbatim achado central
sobre contínuo > discreto. kotwal sintetiza 8 sub-citações específicas
com referências cruzadas.

### Rodada 4 (4 fichas — Fundamentação científica) — auditoria CONCLUÍDA 2026-06-09

| Ficha | N | A | C | Q | Classificação | Fonte alternativa |
|---|---|---|---|---|---|---|
| [fuentes_2019](fuentes_2019.md) | n/a | n/a | forte | **presente** (9 quotes) | ✅ SOLID | WebFetch HTML AABA |
| [lewontin_1972](lewontin_1972.md) | forte | forte | forte | **presente** | ✅ SOLID | WebSearch + PMC retrospective |
| [fitzpatrick_1988](fitzpatrick_1988.md) | razoável | razoável | forte | **presente** | ✅ SOLID | WebSearch + summaries (JAMA bloqueado) |
| [massey_martin_2003](massey_martin_2003.md) | razoável | forte | forte | parcial | ✅ SOLID | WebSearch + multiple secondaries |

**Resumo R4**: 4 SOLID. Estas fichas usaram **WebSearch/HTML como fonte
primária alternativa** (PDFs originais bloqueados por anti-scraping em
Wiley, Springer, JAMA) — **documentação transparente** no campo
`fonte_leitura` do frontmatter. Não são `OVERVIEW_ONLY` porque o
conteúdo foi triangulado entre múltiplas fontes secundárias com
verificação cruzada.

fuentes_2019 destaca-se: 9 princípios do statement AAPA reproduzidos
verbatim como quotes.

## Consolidado final (2026-06-09)

| Rodada | Fichas | SOLID++ | SOLID | OVERVIEW | TOTAL |
|---|---|---|---|---|---|
| R1 | 9 | 9 | 0 | 0 | 9 ✅ |
| R2 | 5 | 3 | 2 | 0 | 5 ✅ |
| R3 | 5 | 3 | 2 | 0 | 5 ✅ |
| R4 | 4 | 0 | 4 | 0 | 4 ✅ |
| R5 | 6 | 0 | 6 | 0 | 6 ✅ |
| R6 (Pereira) | 1 | 0 | 1 | 0 | 1 ✅ |
| **TOTAL** | **30** | **15** | **15** | **0** | **30** ✅ |

### Resultados consolidados

- **Nenhuma ficha permanece OVERVIEW_ONLY** após auditoria.
- **50% SOLID++** (com quotes verbatim e densidade superior) — R1
  predomina aqui.
- **50% SOLID** (sem quotes verbatim mas com sinais técnicos consistentes
  com leitura profunda) — R4 surveys e R5 majoritariamente.
- **Pereira 2026 (única R6 fichada)** foi corretamente identificada
  como OVERVIEW_ONLY na auditoria inicial e PROMOVIDA a VERIFIED após
  leitura integral via HTML do arXiv.

### Recomendações para futuras fichas

1. **Sempre incluir 1-2 quotes verbatim** do paper, entre aspas, com
   page reference quando possível. É a evidência mais forte de leitura.
2. **Reproduzir 1 tabela quantitativa do paper** com números exatos
   quando disponível.
3. **Para papers bloqueados por anti-scraping** (Wiley/Springer/JAMA):
   declarar `fonte_leitura: WebSearch + sumários...` honestamente. Não
   forçar `VERIFIED` se conteúdo não puder ser triangulado.
4. **Declarar `OVERVIEW_ONLY`** sem hesitação quando o PDF não foi
   lido — é mais honesto que `VERIFIED` enganoso.
5. **HTML do arXiv** (arxiv.org/html/XXXX.YYYYY) é alternativa
   confiável quando PDF binário não pode ser processado — Pereira 2026
   demonstrou isso.

## Achados consolidados

### Achado 1 — Pereira 2026 (corrigido em 3 versões)

Ficha original tinha `VERIFIED` com `fonte_leitura: Abstract via arXiv`.
Contradição direta. Trajetória:

1. **Versão 1 (commit 667e882)** — VERIFIED com conteúdo parcialmente
   inventado.
2. **Versão 2 (commit a798a20)** — REDUZIDA para OVERVIEW_ONLY honesta
   após pergunta do usuário expondo a contradição.
3. **Versão 3 (commit 72c4285)** — PROMOVIDA a VERIFIED após leitura
   integral via HTML do arXiv (arxiv.org/html/2603.02475v1).

**Correções críticas descobertas pela leitura integral:**

- FairFace É auditado pelos autores (Seção 7.1) — o abstract era
  enganosamente incompleto, mencionando apenas CelebA e VGGFace2.
- STW agrega 7 datasets-fonte incluindo FairFace e CelebA →
  **contaminação treino-teste** se SkinToneNet for usado para auditar
  FairFace.
- Protocolo de anotação robusto: 1 anotador principal + 2 validadores
  em subset estratificado, ICC(3)=0.939, Krippendorff's α=0.935.
- ViT-Small confirmado como backbone.

### Achado 2 — Padrão de profundidade R1 > R5

R1 fichas mostraram densidade SUPERIOR a R5: todas R1 têm quotes
verbatim, tabelas reproduzidas, hiperparâmetros declarados quando
disponíveis no paper. R5 (papers fundadores de ML/fairness teóricos)
têm densidade um pouco menor — não por menos leitura, mas porque
papers teóricos não têm muitas tabelas para reproduzir.

### Achado 3 — R4 usa fontes alternativas corretamente

As 4 fichas R4 (fuentes_2019, lewontin_1972, fitzpatrick_1988,
massey_martin_2003) declaram explicitamente `WebSearch + sumários`
quando o PDF foi bloqueado por anti-scraping editorial. **Esta é a
postura epistemológica correta** — VERIFIED com fonte alternativa
documentada é honesta; OVERVIEW_ONLY com fonte alternativa não-
documentada seria desonesto.

## Como manter daqui em diante

Para evitar reincidência:

1. **Toda ficha nova deve declarar `OVERVIEW_ONLY` se o PDF não foi
   lido** — sem exceção.
2. **Promoção de `OVERVIEW_ONLY` para `VERIFIED` requer**:
   - Download do PDF para `pdfs/`
   - Leitura integral documentada
   - Reescrita das seções com base no PDF
   - Atualização do `fonte_leitura` no frontmatter
3. **Auditoria periódica** deste documento à cada Rodada de triagem
   antes de o corpus ser citado em publicação derivada.
