---
data: 2026-06-16
tipo: rodada-pesquisa
escopo: literatura sobre diversidade fenotipica intra-Latinx/Hispanic
status: PARCIALMENTE INTEGRADA - 3 fichas criadas (telles_2014, bryc_2015, pew_2017_hispanic_identity)
motivacao: sugestao da analise paralela NotebookLM (2026-06-16) para fortalecer a defesa do gap Latinx F1=60%
data_integracao: 2026-06-16 (Marcello baixou 3 PDFs via VPN/web)
---

> ✅ **STATUS DA RODADA 8**: parcialmente concluída em 2026-06-16.
> Marcello baixou e integrou 3 dos 10 candidatos sugeridos:
> - ✅ `telles_2014` — Pigmentocracies (PERLA Project)
> - ✅ `bryc_2015` — Genetic Ancestry of African Americans, Latinos
>   and European Americans (AJHG)
> - ✅ `pew_2017_hispanic_identity` — Hispanic Identity Fades Across
>   Generations
>
> Estas 3 fichas cobrem o tripé empírico necessário para sustentar
> H3 + C6 com argumento robusto (antropologia + genética + sociologia
> identitária). Os 7 candidatos remanescentes (Bonilla-Silva, Roth,
> Mora, computer vision específicos, AIMs adicionais) permanecem
> opcionais — podem ser adicionados em rodadas futuras se houver
> necessidade, mas argumento já está suficientemente fundamentado.

# Rodada 8 — Diversidade fenotípica intra-Latinx (candidatos para busca)

> **Origem da rodada**: análise paralela do NotebookLM sobre a
> pré-qualificação (2026-06-16) destacou que a categoria
> "Latinx/Hispanic" é tipicamente tratada como bloco monolítico em
> face recognition fairness, sem reconhecer a heterogeneidade
> fenotípica interna — o que contribui para o F1 ≈ 60% persistente.
>
> **Objetivo**: identificar literatura específica que fundamente esta
> heterogeneidade interna e fortaleça a discussão do erro Latinx
> como **componente fenotípico irredutível** (parte central da C6/OE-5).

## 1. Por que esta rodada importa

A literatura computacional atual (`aldahoul_2024`, `lin_2022`,
`karkkainen_2021`) reporta F1 Latinx persistentemente baixo (≈ 60%)
sem explicar a causa. Possíveis explicações:

1. **Heterogeneidade fenotípica interna** — a categoria Latinx
   inclui descendentes europeus, indígenas, africanos e mestiços,
   com espectro MST muito amplo.
2. **Construção sociopolítica da categoria** — "Latinx/Hispanic"
   é classificação social heterodoxa, não taxonomia biológica.
3. **Subamostragem regional** — datasets podem favorecer certos
   subgrupos hispânicos (mexicano-americanos vs caribenhos vs
   sul-americanos).

Esta rodada busca papers que documentem essas heterogeneidades para
sustentar a Hipótese H3 (Latinx spread MST ≥ 5 das 10 classes) e a
Contribuição C6 (decomposição fenotípico × algorítmico).

## 2. Candidatos sugeridos para verificação

### 2.1 Antropologia/sociologia da identidade racial Latina

> ⚠️ **Estes são candidatos teóricos sugeridos com base em
> conhecimento estabelecido** — precisam ser verificados via
> Google Scholar + VPN Unifesp antes de serem promovidos a fichas
> do corpus.

| # | Autor/Trabalho sugerido | Por que considerar |
|---|---|---|
| C1 | **Edward Telles** — "Pigmentocracies: Ethnicity, Race, and Color in Latin America" (UNC Press, 2014) ou trabalhos do **PERLA Project** (Project on Ethnicity and Race in Latin America) | Documenta empíricamente a relação entre tom de pele e identidade racial em sete países latino-americanos — diretamente relevante para MST × Latinx |
| C2 | **Eduardo Bonilla-Silva** — "From Bi-Racial to Tri-Racial: Towards a New System of Racial Stratification in the USA" (Ethnic and Racial Studies, 2004) ou afins | Argumenta que sistema racial USA está caminhando para trinária (White / honorary White / collective Black), com Latinos distribuídos cross categorias |
| C3 | **Wendy Roth** — "Race Migrations: Latinos and the Cultural Transformation of Race" (Stanford UP, 2012) | Análise empírica de como imigrantes latinos negociam classificação racial USA |
| C4 | **G. Cristina Mora** — "Making Hispanics: How Activists, Bureaucrats, and Media Constructed a New American" (Univ Chicago Press, 2014) | História da categoria "Hispanic" no censo USA — construção sociopolítica explícita |
| C5 | **Susan Lopez et al.** — trabalhos sobre identidade racial em populações latinas nos EUA (estudos no Pew Research Center 2015-2024 também relevantes) | Survey-based research sobre auto-identificação racial Hispanic |

### 2.2 Computer vision focando heterogeneidade Latinx

| # | Direção de busca | Termos sugeridos |
|---|---|---|
| C6 | Papers face attribute classification que **decomponham** Hispanic em subgrupos (ex.: mexicano, cubano, sul-americano) | "Hispanic facial recognition heterogeneity", "Latino subgroups face classification" |
| C7 | Papers que documentem **spread fenotípico** em datasets de FR para grupos Latinos | "FairFace Latinx phenotype spread", "Hispanic skin tone variance" |
| C8 | Datasets brasileiros/latino-americanos específicos de FR | "FACET dataset", "Brazilian face dataset", "Latin American FR benchmark" |

### 2.3 Genética populacional de populações americanas mestiças

| # | Trabalho sugerido | Por que considerar |
|---|---|---|
| C9 | **Ancestry-informative markers (AIMs)** em populações latino-americanas — trabalhos de Bryc, Bustamante, Sandoval-Velasco | Quantifica empíricamente proporções African/European/Indigenous em populações latinas — ponte direta entre Lewontin 1972 e nosso achado MST × race |
| C10 | **23andMe / Ancestry research papers** sobre composição genética latino-americana | Demonstra heterogeneidade interna mensurável em escala populacional |

## 3. Critérios de seleção (R8)

Para promover a ficha do corpus principal, candidatos devem:
- Ter venue peer-reviewed (não blog/popular press)
- Ser citáveis em ABNT
- Adicionar argumento NÃO redundante com o já presente em
  [[fuentes_2019]] (AAPA Statement) ou [[lewontin_1972]]
- Ter PDF acessível via VPN Unifesp ou Open Access

**Meta**: ≤ 5 fichas promovidas — esta é rodada **enxuta**, focada
exclusivamente no gap Latinx.

## 4. Plano de busca recomendado

1. **Google Scholar** com termos:
   - "phenotypic heterogeneity AND (Latino OR Hispanic) AND
     (face recognition OR computer vision)"
   - "skin tone variance AND Latinx AND fairness"
   - "Hispanic identity AND race classification AND machine learning"
2. **Web of Science** para verificar venue e citações
3. **PERLA project** site oficial para Telles + colaboradores
4. **arXiv cs.CV** para preprints recentes 2024-2026

## 5. Tratamento na tese (mesmo se busca não encontrar nada novo)

A análise crítica desta rodada já gerou valor:

### 5.1 Adições à narrativa (mesmo sem novas fichas)

A pré-qualificação narrativa deve incluir explicitamente — fundamentado
pelos clássicos já no corpus — que:

> *"A categoria 'Latinx/Hispanic' do FairFace constitui rótulo
> sociopolítico de heterogeneidade fenotípica conhecida, agregando
> descendentes de múltiplas ancestralidades (europeia, indígena,
> africana e mestiça). Esta heterogeneidade interna, documentada na
> literatura antropológica (Fuentes et al. 2019; Lewontin 1972), é
> hipotetizada nesta dissertação como contribuição estrutural ao
> baixo F1 reportado para Latinx em modelos do estado-da-arte
> (60%, AlDahoul et al. 2024)."*

### 5.2 Reforço metodológico

Adicionar à H3 (matriz MST × race) análise explícita do **spread MST
dentro da categoria Latinx** — se confirmar spread ≥ 5 das 10
classes MST, valida empiricamente a hipótese de heterogeneidade.

### 5.3 Argumentação ética

Conectar com Fuentes et al. (2019, AAPA Statement) que afirma "no
group of people is, or ever has been, biologically homogeneous or
'pure'" — aplicado especificamente ao caso Latinx no FairFace.

## 6. Status

- **Esta rodada NÃO é bloqueante** para a escrita do Cap 1.
- Pode ser executada em paralelo durante semanas 2-4 do plano de
  escrita.
- Se nenhum candidato sobreviver à triagem, a Seção 5 acima continua
  válida como argumento da tese — fundamentada nos clássicos já
  no corpus.

## 7. Próximas ações

- [ ] Marcello: executar busca via Google Scholar + VPN
- [ ] Triar candidatos contra critérios da Seção 3
- [ ] Criar até 5 fichas OVERVIEW_ONLY
- [ ] Promover a VERIFIED após leitura
- [ ] Atualizar `_mapa_citacoes_por_capitulo.md` se promovidas
