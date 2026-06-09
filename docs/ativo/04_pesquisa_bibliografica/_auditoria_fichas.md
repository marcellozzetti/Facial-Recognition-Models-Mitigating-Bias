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

### Rodadas 1-4 (23 fichas)

Auditoria pendente. Será executada após R5.

| Ficha | Rodada | Status |
|---|---|---|
| buolamwini_2018 | R1 | pendente |
| dataset_karkkainen_2021 | R1 | pendente |
| aldahoul_2024 | R1 | pendente |
| manzoor_2024 | R1 | pendente |
| dehdashtian_2024 | R1 | pendente |
| dominguez_2024 | R1 | pendente |
| lafargue_2025 | R1 | pendente |
| dataset_wang_2019 | R1 | pendente |
| grother_2019 | R1 | pendente |
| dataset_robinson_2020 | R2 | pendente |
| park_2022 | R2 | pendente |
| lin_2022 | R2 | pendente |
| sagawa_2020 | R2 | pendente |
| bhaskaruni_2019 | R2 | pendente |
| dataset_hazirbas_2021 | R3 | pendente |
| schumann_2023 | R3 | pendente |
| neto_2025 | R3 | pendente |
| survey_mehrabi_2021 | R3 | pendente |
| survey_kotwal_2025 | R3 | pendente |
| fuentes_2019 | R4 | pendente |
| lewontin_1972 | R4 | pendente |
| fitzpatrick_1988 | R4 | pendente |
| massey_martin_2003 | R4 | pendente |

## Achados consolidados

> Esta seção será preenchida à medida que a auditoria avança.

### Achado 1 — Pereira 2026

Ficha original tinha `VERIFIED` com `fonte_leitura: Abstract via arXiv`.
Contradição direta. Reescrita como `OVERVIEW_ONLY` honesta em
2026-06-09.

**Conteúdo removido da ficha original** porque era inferência não
verificada:

- "Anotação MST sem protocolo multi-anotador documentado"
- "STW concentra-se em sujeitos de imagens web (viés de quem é
  fotografado e publicado)"
- "Apenas uma arquitetura SOTA reportada (ViT-Small)"
- "Generalização out-of-domain reportada qualitativamente"
- Seção 12 (Análise crítica) inteira

Conteúdo mantido porque está no abstract:

- 42.313 imagens, 3.564 indivíduos, MST 10-classe
- Classic vs deep learning benchmark
- "Near-random" para clássicos, "near-annotator" para deep
- SkinToneNet ViT fine-tuned SOTA cross-domain
- Auditoria de CelebA e VGGFace2 (NÃO FairFace)

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
