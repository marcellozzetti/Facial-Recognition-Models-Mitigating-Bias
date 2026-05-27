# Pesquisa Bibliográfica — metodologia e operação

> Documento normativo. Estabelece o processo de pesquisa bibliográfica
> da dissertação: critérios de seleção de papers, fluxo de trabalho,
> template de ficha por paper e convenções de catalogação. Todas as
> fichas neste diretório seguem o template definido aqui.
>
> Criado em 2026-05-25 como parte da reestruturação documental pós-pivot.

## 1. Objetivo da Pesquisa Bibliográfica

Mapear, com rigor metodológico, o estado da arte de **equidade
demográfica (fairness) em biometria facial e classificação racial
facial**, identificando lacunas (gaps) que possam fundamentar
contribuição original ao campo.

Saídas esperadas (em arquivos próprios, fora deste diretório):

- `05_landscape.md` — síntese da paisagem da literatura (tendências,
  abordagens dominantes, datasets, métricas, venues).
- `06_gap.md` — identificação dos gaps reais, com sustentação na
  evidência catalogada.
- `07_thesis_statement.md` (v3) — thesis statement reformulado sobre o
  gap identificado, não sobre evolução do MBA.

## 2. Princípio operacional

**Toda afirmação sobre um paper, em qualquer documento ativo da
dissertação, deve poder ser rastreada até uma ficha aqui catalogada.**
Sem ficha, sem citação. Síntese sobre abstract não conta como leitura.

A ficha resulta de **leitura integral do PDF**, não de fetch de
metadados ou parsing de abstract.

## 3. Critérios de seleção

Aplicados nesta ordem, em duas etapas (triagem editorial → leitura
integral).

### 3.1 Critérios de inclusão (obrigatórios)

1. **Escopo temático** — o paper aborda pelo menos um destes eixos:
   (a) fairness/equidade em biometria facial;
   (b) classificação racial facial (in-domain ou cross-dataset);
   (c) mitigação de viés demográfico em CNN/Transformer faciais;
   (d) auditoria/medição de disparidade demográfica em sistemas
   biométricos faciais.

2. **Recência** — publicado em **2019–2026** (10 anos pós-Buolamwini &
   Gebru 2018, marco fundador). Exceções: papers seminais anteriores
   citados como referência metodológica fundamental.

3. **Verificabilidade** — autoria, ano e venue confirmáveis em fonte
   primária: arXiv, DOI editorial, página oficial do venue. Sem fonte
   primária, fora.

### 3.2 Critério de relevância editorial

4. **Venue revisado por pares** — preferência por:
   - **Visão computacional / ML**: CVPR, ICCV, ECCV, NeurIPS, ICML,
     ICLR, AAAI, WACV, BMVC.
   - **Fairness / sociotécnico**: ACM FAccT (ex-FAT*), AAAI/ACM AIES,
     EAAMO.
   - **Biometria**: IJCB, BTAS, IEEE TBIOM, IEEE TIFS.
   - **Periódicos top**: IEEE TPAMI, IJCV, ACM Computing Surveys,
     Information Fusion, Pattern Recognition.
   - **Preprints arXiv**: aceitos apenas se atenderem ao critério 5
     (citações) OU cobrirem aspecto único não coberto por papers
     publicados.

### 3.3 Critério de impacto (modulado por idade)

5. **Contagem de citações** — via Google Scholar **e** Semantic Scholar
   (cross-check):
   - **Papers ≥ 3 anos** (até 2023): exigir **≥ 50 citações** para
     inclusão automática; entre 10–50 apenas se cobertura única e
     justificada.
   - **Papers de 2024**: exigir **≥ 20 citações** para inclusão
     automática; abaixo disso, justificar cobertura.
   - **Papers de 2025–2026**: contagem ainda imatura; aceitar se venue
     forte (critério 4) **e** proximidade temática alta.
   - **Surveys**: threshold mais alto — **≥ 100 citações** se ≥ 3 anos.
     Surveys pouco citados em geral refletem baixa qualidade ou nicho
     estreito.

### 3.4 Critérios de exclusão

- Papers sem fonte primária verificável.
- Tutoriais, blog posts, white papers corporativos sem revisão por pares.
- Trabalhos estritamente de identificação 1:1 (face matching) sem
  componente demográfica.
- Datasets sem paper acompanhante (dataset notes em sites institucionais
  não substituem paper).
- Citações secundárias usadas como se fossem leitura primária.

### 3.5 Lente de "disrupção" (para identificação de gaps)

Ao avaliar cada paper, registrar oportunidade em uma das 3 lentes:

- **Cobertura** — categorias demográficas, datasets ou cenários não
  cobertos pela literatura existente (ex.: raça em 7 categorias
  in-domain pouco explorada).
- **Metodológica** — técnicas combinadas ou aplicadas de forma não
  documentada (ex.: ensemble profundo + calibração de temperatura sob
  protocolo casado para fairness).
- **Paradigma** — reframing do problema (ex.: tratar fairness como
  problema de calibração condicional, não de equalização de métricas).

## 4. Fluxo de trabalho

```
candidato → triagem editorial (crit. 1–5) → leitura integral → ficha catalogada
                  ↓ (reprovado)
            _descartados.md (com justificativa)
```

### Fase A — Coleta sistemática (queries)

Fontes obrigatórias (em ordem):

1. **Google Scholar** — queries:
   - "fairness facial recognition" + ano filtrado.
   - "racial bias face classification".
   - "demographic disparity face recognition".
   - "FairFace" (dataset-anchored).
2. **Semantic Scholar** — busca por autor + "cited by" dos seeds.
3. **arXiv** — categorias `cs.CV`, `cs.LG`, `cs.CY` com filtro de palavras.
4. **DBLP** — para autores recorrentes (snowballing).
5. **Proceedings dos venues do critério 4** — varredura por título.

Toda query registrada em `_queries_log.md` com data, fonte e nº de hits
revistos.

### Fase B — Triagem editorial

Cada candidato passa por checagem de:
- Autoria + ano + venue (critérios 1–4).
- Contagem de citações em Google Scholar + Semantic Scholar (critério 5).
- Decisão registrada em `_triagem.md`: ✅ aprovado para leitura | ❌ descartado | ⚠ standby.

### Fase C — Leitura integral e ficha

Para cada paper aprovado:
- Baixar PDF para `pdfs/<autor>_<ano>.pdf` (gitignored).
- Ler integralmente.
- Preencher ficha `<autor>_<ano>.md` seguindo o template (Seção 5).
- Atualizar `00_referencias.md` se faltar entrada VERIFIED.

### Fase D — Síntese cruzada

Quando o corpus catalogado atingir massa crítica (estimativa: 20–30
fichas), abrir `05_landscape.md` e cruzar achados por:
- Datasets dominantes.
- Métricas reportadas.
- Estratégias de mitigação (in-/pre-/post-processing).
- Lacunas recorrentes.

## 5. Template de ficha por paper

Cada ficha deve seguir esta estrutura, salva como
`<primeiro_autor_minusculo>_<ano>.md`:

```markdown
---
name: <primeiro_autor>-<ano>
status_verificacao: VERIFIED   # VERIFIED | CONTENT-TO-VERIFY
autores: [Lista, Completa, Aqui]
ano: YYYY
titulo: "Título exato do paper"
venue: "Nome completo do venue (acrônimo)"
tipo_publicacao: conference   # conference | journal | preprint | book_chapter
arxiv_id: XXXX.XXXXX          # ou null
doi: 10.XXXX/...              # ou null
url_primario: https://...
citacoes_google_scholar: N
citacoes_semantic_scholar: N
data_verificacao_citacoes: YYYY-MM-DD
n_referencias_paper: N
lente_disrupcao: cobertura | metodologica | paradigma | nenhuma
---

## 1. Resumo do problema atacado
[2–4 linhas: que problema o paper resolve, em quais termos.]

## 2. Método
[Descrição da técnica proposta. Componentes principais. O que é novo
vs o que é herdado de trabalhos anteriores.]

## 3. Datasets e setup experimental
[Quais datasets, partições, baselines, hardware, hiperparâmetros
relevantes. Anotar se usa partição oficial ou própria.]

## 4. Métricas reportadas
[Lista exata: accuracy, F1, DR, equalized odds, etc. Com valores
numéricos centrais (tabela curta se for o caso).]

## 5. Resultados principais (valores numéricos)
[Quadro consolidado com os números que importam. Sem arredondar de
forma a perder o efeito.]

## 6. Limitações declaradas pelos autores
[Cópia ou paráfrase fiel da seção "Limitations" / "Future work".]

## 7. Limitações que identifiquei (leitura crítica)
[O que o paper omite, simplifica indevidamente, ou onde a evidência
não sustenta a conclusão.]

## 8. Relação com nossa pesquisa
[Como este paper se conecta com o que estamos investigando. É baseline?
Anchor? Negativo (mostra que algo já foi tentado e falhou)? Posiciona o
nosso gap?]

## 9. Pontos para citar / posicionar
[Frases curtas que poderiam virar citação no texto da dissertação.
Não copiar trechos do paper — formular paráfrases verificáveis.]

## 10. Arquivos relacionados
- PDF local: `pdfs/<autor>_<ano>.pdf`
- Entradas relacionadas: [[outro-autor-ano]], [[outro-autor-ano]]
- Referência canônica em: `docs/ativo/00_referencias.md` (Seção X.Y)

## 11. Trabalhos sugeridos pelos autores (Future Work)
[Lista literal das direções de "Future Work" / "Limitations
addressable in future" / "Conclusions" do paper, com referência ao
local no PDF (página/seção). Marcar cada item com:
- ✅ Alinhada com nossa frente 🔬 (citar Q-id)
- ⚠ Parcialmente alinhada
- ❌ Divergente / fora do nosso escopo]
```

## 6. Convenções de nomenclatura

- **Nome do arquivo**: `<primeiro_autor_minusculo>_<ano>.md` (ex.:
  `buolamwini_2018.md`, `aldahoul_2024.md`).
- **Colisão de ano/autor**: sufixo `_a`, `_b` (ex.:
  `manzoor_2024_a.md`).
- **Surveys**: prefixo `survey_` (ex.: `survey_caton_2024.md`).
- **Datasets**: prefixo `dataset_` (ex.: `dataset_karkkainen_2021.md`
  para FairFace).
- Toda referência cruzada entre fichas usa `[[nome-do-arquivo-sem-md]]`.

## 7. Papers seed (ponto de partida)

Os 7 papers já verificados em `00_referencias.md` (Seção 1+2+3) são os
candidatos primários a ficha. Antes de gerar ficha, cada um passa pela
**triagem editorial** (critérios 4 e 5):

| Paper | Triagem prevista |
|---|---|
| Buolamwini & Gebru 2018 (Gender Shades) | ✅ FAT* + alta citação |
| Karkkainen & Joo 2021 (FairFace) | ✅ WACV + alta citação |
| AlDahoul et al. 2024 (VLM facial attributes) | ⚠ verificar venue/citações |
| Manzoor & Rattani 2024 (FineFACE) | ⚠ verificar venue/citações |
| Dehdashtian, Sadeghi, Boddeti 2024 (U-FaTE) | ⚠ verificar venue/citações |
| Dominguez-Catena, Paternain, Galar 2024 (DSAP) | ⚠ verificar venue/citações |
| Lafargue, Claeys, Loubes 2025 | ⚠ recente, depende de cobertura única |

Papers seed adicionais a buscar (não verificados ainda):
- **RFW dataset** (Wang et al. — Racial Faces in the Wild).
- **NIST FRVT Demographic Effects** (Grother et al. — relatório NIST).
- **Fair face generation** (ex.: trabalhos com GAN/diffusion para
  balanceamento demográfico).

## 8. Arquivos administrativos deste diretório

- `README.md` — este arquivo (metodologia).
- `_queries_log.md` — registro de queries executadas (Fase A).
- `_triagem.md` — registro da triagem editorial (Fase B).
- `_descartados.md` — papers descartados com justificativa.
- `_perguntas.md` — **registro de perguntas de pesquisa vs respostas**
  agregadas do corpus (ver §9).
- `pdfs/` — PDFs baixados (gitignored).
- `<autor>_<ano>.md` — uma ficha por paper aprovado.

## 9. Perguntas de pesquisa (Q&A metodológico)

Mecanismo para transformar a leitura **passiva** das fichas em
**interrogação ativa** da literatura. Cada pergunta de pesquisa é
respondida via **agregação de evidências** do corpus catalogado em
`<autor>_<ano>.md`. Se o corpus não responde, a pergunta vira **nova
frente de pesquisa** explícita.

### 9.1 Estrutura

Todas as perguntas são registradas em `_perguntas.md` com o template:

```markdown
## Q<NN> — <pergunta>

- **Status:** ✅ ANSWERED | ⚠ PARTIAL | ❌ OPEN | 🔬 NEW RESEARCH FRONT
- **Data:** YYYY-MM-DD (primeira investigação)
- **Última atualização:** YYYY-MM-DD
- **Fichas consultadas:** [[autor-ano]], [[autor-ano]], ...

### Evidências coletadas
[Síntese das evidências por ficha, com citação numérica:
  - [[autor-ano]] reporta X (Seção Y da ficha).]

### Resposta
[Síntese consolidada da resposta.]

### Lacunas / Nova frente de pesquisa
[O que NÃO foi respondido. Se gera direção experimental concreta,
marcar 🔬 NEW RESEARCH FRONT com descrição da hipótese a investigar.]
```

### 9.2 Status flags

- **✅ ANSWERED:** corpus oferece resposta consolidada e cross-validated
  por ≥2 fichas independentes.
- **⚠ PARTIAL:** resposta existe mas com cobertura limitada (e.g., só
  um paper aborda; cobertura temporal incompleta).
- **❌ OPEN:** nenhuma ficha do corpus aborda a pergunta. Pode
  indicar (a) necessidade de mais snowballing, (b) gap genuíno na
  literatura.
- **🔬 NEW RESEARCH FRONT:** a pergunta, ao não ter resposta, define
  uma direção experimental para a dissertação. Deve ser referenciada
  em `06_gap.md`.

### 9.3 Fluxo de uso

1. **Antes de iniciar uma síntese** (e.g., `05_landscape.md`,
   `06_gap.md`), listar as perguntas que precisam ser respondidas.
2. **Para cada pergunta**, consultar fichas relevantes e sintetizar
   resposta em `_perguntas.md`.
3. **Se nova evidência surgir** (nova ficha, paper novo), atualizar
   o registro da pergunta com "Última atualização" e revisar status.
4. **Perguntas com status 🔬** alimentam `06_gap.md` diretamente.

### 9.4 Vínculo com fichas

Opcionalmente, fichas podem adicionar uma **Seção 11 — Perguntas
que este paper ajuda a responder** com lista de Q<NN> ids. Não é
obrigatório nesta etapa; preferimos manter o catálogo bidirecional em
`_perguntas.md` (perguntas referenciam fichas; fichas referenciam
perguntas só quando intuição de discoverability justifica).

## 10. Pendências

- [ ] Re-verificar venue + citações dos 7 papers seed acima.
- [ ] Buscar papers seed adicionais (RFW, NIST FRVT, fair generation).
- [ ] Criar `_queries_log.md`, `_triagem.md`, `_descartados.md` quando
      iniciar Fase A.
- [ ] Definir threshold de massa crítica para abrir `05_landscape.md`
      (provisório: 20–30 fichas).
