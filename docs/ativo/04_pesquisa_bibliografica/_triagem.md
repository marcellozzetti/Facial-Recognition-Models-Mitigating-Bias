# Triagem editorial — log de decisões

> Registro da Fase B do fluxo definido em `README.md` §4. Cada paper
> candidato passa por checagem de critérios 1–5 (venue + citações) antes
> de gerar ficha completa. Decisões: ✅ aprovado | ❌ descartado | ⚠ standby.

## Convenções

- **Data de verificação**: registrar para cada linha (citações mudam).
- **Citações**: prioridade Semantic Scholar API (cross-check com Google
  Scholar quando viável). Quando API falha por rate limit, marcar
  origem alternativa.
- **Standby**: paper que viola um critério quantitativo (geralmente
  citações abaixo do threshold) mas é aprovado por exceção
  documentada (cobertura única, venue alternativo forte, proximidade
  temática crítica).

## Rodada 1 — Seeds iniciais (2026-05-25)

### 1.1 Papers seed já listados em `00_referencias.md`

| # | Paper | Venue | Tipo | Citações | Decisão | Justificativa |
|---|---|---|---|---|---|---|
| S1 | **Buolamwini & Gebru (2018)** — Gender Shades | PMLR vol 81 / ACM FAT* | conference (peer-reviewed) | **4 933** (Semantic Scholar via search 2026-05-25) | ✅ APROVADO | Venue forte + impacto seminal (>4 000 citações). Atende critérios 1–5 integralmente. |
| S2 | **Karkkainen & Joo (2021)** — FairFace | WACV 2021 (IEEE/CVF) | conference (peer-reviewed) | ≥263 (Google Scholar via search 2026-05-25, subestimado; API S2 não consultada por rate limit) | ✅ APROVADO | Venue forte + dataset central do trabalho (paper-anchor). Re-verificação numérica pendente, mas decisão sustentada pela centralidade. |
| S3 | **AlDahoul, Tan, Kasireddy & Zaki (2024)** — Exploring VLMs for Facial Attribute Recognition | arXiv 2410.24148 (preprint) | preprint | **11** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Não publicado em venue peer-reviewed; abaixo do threshold de 2024 (≥20). MAS é a **única referência publicada para a tarefa exata** que rodamos (raça 7-class no-domain via VLM). Aprovar por **cobertura única** (critério 4 alternativo, §3.2 do README). Acompanhar publicação follow-up dos autores em Nature Scientific Reports (FaceScanPaliGemma, 2026 — verificar separadamente). |
| S4 | **Manzoor & Rattani (2024)** — FineFACE | **ICPR 2024** (Springer LNCS) | conference (peer-reviewed) | **2** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Venue OK (ICPR é peer-reviewed). Citações abaixo do threshold de 2024 (≥20). Aprovar por proximidade temática alta (fair facial attribute classification) + necessidade de posicionamento crítico (o paper foi inicialmente confundido com classificação racial; auditoria revelou que classifica gênero). É um caso paradigmático de **leitura crítica vs leitura de abstract**. |
| S5 | **Dehdashtian, Sadeghi & Boddeti (2024)** — U-FaTE | **CVPR 2024** | conference (peer-reviewed) | **23** (Semantic Scholar 2026-05-25) | ✅ APROVADO | Venue top da área + acima do threshold de 2024 (≥20). Atende critérios 1–5 integralmente. |
| S6 | **Dominguez-Catena, Paternain & Galar (2024)** — DSAP | **Information Fusion** (Elsevier, journal) | journal (peer-reviewed) | **7** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Periódico top da área (Information Fusion, IF~14). Citações abaixo do threshold MAS venue forte compensa (critério 4 prevalece sobre 5 quando periódico top). Nota: registro Semantic Scholar lista year=2023; arXiv submissão dez/2023; publicação efetiva em Information Fusion 2024. Resolver discrepância na ficha. |
| S7 | **Lafargue, Claeys & Loubes (2025)** — Fairness is in the Details | **ECML PKDD 2025** (Springer LNCS) | conference (peer-reviewed) | **1** (Semantic Scholar 2026-05-25) | ⚠ STANDBY → ✅ APROVADO POR EXCEÇÃO | Venue OK. Paper muito recente (2025), citações naturalmente baixas. Aprovar por **cobertura única**: única referência identificada que ancora auditoria de fairness na regulação europeia (EU AI Act). |

### 1.2 Papers seed adicionais (a incorporar em `00_referencias.md`)

| # | Paper | Venue | Tipo | Citações | Decisão | Justificativa |
|---|---|---|---|---|---|---|
| S8 | **Wang, Deng, Hu, Tao & Huang (2019)** — Racial Faces in-the-Wild (RFW) | **ICCV 2019** (IEEE/CVF) | conference (peer-reviewed) | a verificar (estimado alto — dataset seminal) | ✅ APROVADO | Venue top + dataset seminal complementar ao FairFace para auditoria de bias em face recognition (não classification). Permite triangulação. |
| S9 | **Grother, Ngan & Hanaoka (2019)** — NISTIR 8280 / FRVT Part 3: Demographic Effects | **NIST Interagency Report** (gov technical report) | technical report | n/a (não-paper; relatório oficial) | ✅ APROVADO POR EXCEÇÃO | Relatório técnico oficial NIST com auditoria industry-wide de ~200 algoritmos e ~18M imagens. Não revisado por pares no sentido acadêmico, mas é a **única fonte regulatória/industry com escala desse porte**. Aprovar como evidência industry-wide complementar à literatura acadêmica. |
| S10 | **Candidatos a fair generation** (Friedrich et al. "Gaussian Harmony" arXiv 2312.14976; Perera & Patel "Unbiased-Diff"; outros) | arXiv preprints / Springer | preprint / chapter | a verificar | ⚠ STANDBY | Aspecto **periférico** à nossa pesquisa (geração vs classificação). Decidir inclusão após `06_gap.md`: se o gap envolver dimensão generativa, incorporar; se não, descartar para `_descartados.md` com justificativa. |

## 2. Resumo da Rodada 1

- **Aprovados diretamente (critérios atendidos sem exceção):** S1, S2, S5, S8 — 4 papers.
- **Aprovados por exceção (cobertura única ou venue forte compensando):** S3, S4, S6, S7, S9 — 5 papers.
- **Standby para decisão pós-gap:** S10 (fair generation) — 1 grupo.
- **Descartados:** nenhum nesta rodada.

**Total aprovado para Fase C (leitura integral + ficha):** 9 papers.

## 3. Pendências de verificação numérica

Itens que ficaram com dados parciais por rate limit ou ainda não buscados:

- [ ] Citação Semantic Scholar de **FairFace** (Karkkainen & Joo 2021, arXiv:1908.04913) — API retornou 429.
- [ ] Citação Semantic Scholar de **RFW** (Wang et al. 2019, arXiv:1812.00194) — não consultada.
- [ ] Verificar se **FaceScanPaliGemma** (Nature Scientific Reports 2026, follow-up de AlDahoul et al.) deve entrar como S3-bis.
- [ ] Decidir candidato representativo único para S10 (fair generation) ou descartar grupo inteiro pós-gap.

## 4. Próxima ação

Iniciar **Fase C** — leitura integral e ficha — pelos 4 papers aprovados sem exceção (S1, S2, S5, S8) para construir baseline metodológico. Em seguida, os 5 papers aprovados por exceção.

Ordem sugerida de leitura (priorização por valor para `06_gap.md`):

1. **S2 FairFace** — dataset central, fundamenta toda a discussão.
2. **S1 Gender Shades** — marco fundador do campo, ancora a motivação ética.
3. **S5 U-FaTE** — formaliza o trade-off utility–fairness, esqueleto teórico.
4. **S8 RFW** — segundo dataset de bias auditing, permite triangulação.
5. **S6 DSAP** — metodologia de auditoria de datasets, conecta com S9.
6. **S9 NISTIR 8280** — escala industrial, ancora positioning.
7. **S3 AlDahoul et al.** — única referência para a tarefa exata.
8. **S4 FineFACE** — armadilha textual já identificada (gênero ≠ raça); ler para entender o que NÃO é nosso problema.
9. **S7 Lafargue et al.** — auditoria regulatória, ancora discussão ética/EU AI Act.
