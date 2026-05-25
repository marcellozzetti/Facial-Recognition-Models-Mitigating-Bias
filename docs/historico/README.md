# Histórico — material anterior à reestruturação de 2026-05-25

> **Atenção:** os arquivos deste diretório representam o estado da
> documentação **antes** do pivot estratégico decidido na reunião com
> o orientador em 25 de maio de 2026. Eles são preservados para
> rastreabilidade temporal e auditoria do processo de pesquisa, mas
> **não devem ser citados diretamente em novos materiais sem revisão
> integral**.

## Contexto da reestruturação

Em 2026-05-25, na reunião de progresso com o orientador (Prof. Marcos
Quiles), foram identificados três problemas estruturais no material
até então produzido:

1. **Pesquisa científica considerada rasa** — extrações superficiais
   sobre abstracts foram tratadas como leitura aprofundada de artigos,
   resultando em sínteses que não se sustentaram quando questionadas
   em profundidade.

2. **Citações com autoria incorreta** — verificação posterior em fonte
   primária (arXiv) revelou que 5 de 7 papers principais haviam sido
   atribuídos a autores errados. Especificamente:
   - "Hassanpour et al. 2024" → realmente **AlDahoul, Tan, Kasireddy, Zaki, 2024**
   - "Liu et al. 2024" (FineFACE) → realmente **Manzoor & Rattani, 2024**
   - "Sojitra et al. 2024" (U-FaTE) → realmente **Dehdashtian, Sadeghi, Boddeti, 2024**
   - "Sánchez-Sánchez et al. 2024" (DSAP) → realmente **Dominguez-Catena, Paternain, Galar, 2024**
   - "Galera-Zarco et al. 2025" (Fairness-in-Details) → realmente **Lafargue, Claeys, Loubes, 2025**

3. **Framing fundado em "evolução do MBA"** — orientador recomendou
   reformular a tese sobre uma lacuna identificada na literatura, e
   não sobre a evolução incremental do pipeline herdado do MBA-USP.

## O que foi feito antes da preservação

Antes de mover os arquivos para este diretório, as **5 citações
incorretas listadas acima foram corrigidas** via substituição textual
em todos os arquivos `.md` afetados (172 substituições em 13 arquivos).
Assim, mesmo o conteúdo arquivado aqui não propaga as autorias
incorretas originais.

Permanecem como possíveis problemas (não corrigidos no histórico):

- **Outras citações ainda não verificadas** (papers de fundamentação
  metodológica como Lakshminarayanan 2017, Pineau 2021, Bhaskaruni 2019
  etc.) podem ter erros similares.
- **Conteúdo sintetizado de papers a partir de abstracts** pode estar
  parcialmente incorreto na descrição de métodos, resultados ou
  posicionamento — apenas as autorias foram saneadas.
- **Framing "evolução do MBA"** está presente em vários documentos e
  precisa ser reformulado nos materiais ativos.

## Política de uso deste diretório

- **Permitido**: consultar para reconstruir o histórico de decisões,
  ver registro de experimentos rodados, recuperar trechos de análise
  para reuso após re-verificação.
- **Não permitido**: copiar texto diretamente para `ativo/` sem
  re-verificação completa (autoria + conteúdo do paper citado).
- **Citações ao reuso**: se um trecho de `historico/X.md` for usado em
  `ativo/`, indicar explicitamente a origem e o que foi re-verificado.

## Inventário (alto nível)

Arquivos preservados aqui incluem:

- **Estratégia e narrativa anterior**: `THESIS_STATEMENT.md` (v1 e v2),
  `PROPOSTA_MESTRADO.md`, `PLANO_TRABALHO.md`, `disruption_roadmap.md`,
  `nomenclatura_experimentos.md`.
- **Resultados experimentais**: `factor3_results.md`, `factor4_results.md`,
  `factor5_results.md`, `anchors_results.md`, `combo_defesa_fechamento.md`,
  `intersectional_analysis.md`, `auditoria_codigo_limitadores.md`,
  `r2_clean_dataset_results.md`, `r4_refit_results.md`,
  `hpo_round1_results.md`, `hpo_round2_results.md`, etc.
- **Síntese de literatura** (com problemas conhecidos, mesmo após
  correção de autorias): `sota_papers_summary.md`, `sota_pdf_synthesis.md`,
  `sota_7class_race_audit.md`, `sota_review.md`, `baseline_positioning.md`,
  `formula_desk_check.md`, `fairface_references_analysis.md`,
  `literature_semantic_audit.md`, `literature_corpus.csv`,
  `literature_corpus_abstracts.csv`.
- **Diagnósticos e auditorias**: `checkpoint_criterion_audit.md`,
  `magface_diagnosis.md`, `dataset_audit_findings.md`, `security_audit.md`.
- **Material de reuniões**: `meeting_2026-05-11_kickoff.md`,
  `meeting_prep_2026-05-11.md`, `meeting_prep_2026-05-18.md`,
  `status_orientador_2026-05-18.md`, `apresentações_arquivo/`.
- **Trabalhos preliminares (origem MBA)**: `exp01_vs_mba.md`,
  `clean_results.md`, `dataset_factor_results.md`, `smoke_results.md`,
  `HISTORICO.md`, `GLOSSARIO.md`.
- **Comprovantes acadêmicos**: `requerimento_estudos_dirigidos.md`,
  `slides_qualificacao.md`.

## Data da reestruturação

25 de maio de 2026.
