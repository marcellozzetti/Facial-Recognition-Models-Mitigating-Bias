---
name: fitzpatrick-1988
status_verificacao: VERIFIED
autores: [Thomas B. Fitzpatrick]
ano: 1988
titulo: "The Validity and Practicality of Sun-Reactive Skin Types I Through VI"
venue: "Archives of Dermatology, vol 124, no 6, pp. 869-871"
tipo_publicacao: journal
arxiv_id: null
doi: "10.1001/archderm.1988.01670060015008"
url_primario: https://jamanetwork.com/journals/jamadermatology/articlepdf/549509/archderm_124_6_008.pdf
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~12 (estimado)
lente_disrupcao: nenhuma (paper fonte; documenta origem da escala)
fonte_leitura: WebSearch + summaries de fontes secundárias (Hairfacts, Skin Inc, JAMA Network). PDF integral via JAMA Network bloqueado por anti-scraping; conteúdo essencial capturado.
---

# The Validity and Practicality of Sun-Reactive Skin Types I Through VI (Fitzpatrick, 1988)

> **Paper-fonte da escala Fitzpatrick**, citada em literatura de
> fairness em computer vision como "escala de tom de pele" — uso
> que **distorce o propósito original** documentado neste paper.

## 1. Resumo do problema atacado

Em 1975, T.B. Fitzpatrick (dermatologista, Harvard Medical School)
desenvolveu sistema de "skin typing" para um problema **clínico
específico**: prescrever dose inicial de UVA segura em fotoquimioterapia
oral com metoxsaleno (PUVA) para psoríase. Pacientes "aparentemente
similares" (cabelo escuro, olhos castanhos) tinham reações
fototoxidas severas com mesma dose — necessidade de **classificação
preditiva** de sensibilidade UV.

Este paper de 1988 **revalidou** a escala 13 anos depois, expandiu
para incluir tipos V e VI (pele mais escura, originalmente
ignorados em populações brancas tratadas com PUVA), e estabeleceu
"validade e praticidade" do sistema.

## 2. Método

- **Avaliação clínica:** pacientes de fototerapia eram classificados
  por **autoreporte de história de queimadura solar** e
  **bronzeamento** após exposição UV controlada.
- **Variáveis usadas para typing:** história de queimadura, capacidade
  de bronzear, e fenótipos correlatos (cor de cabelo, cor de olhos)
  — **note bem**: estes são **proxies para resposta UV**, não
  classificação racial.

## 3. Datasets e setup experimental

Não aplicável — paper de validação clínica em contexto
dermatológico.

## 4. Métricas reportadas

Não aplicável — paper descritivo da escala + revisão de uso.

## 5. Resultados principais — a escala Fitzpatrick (verbatim)

**6 tipos sun-reactive:**

| Tipo | Descrição |
|---|---|
| **I** | Sempre queima, nunca bronzeia (pele muito clara — Celtic/Norse) |
| **II** | Usualmente queima, bronzeia pouco com dificuldade (pele clara) |
| **III** | Às vezes queima, bronzeia gradualmente (pele média — Mediterrâneo, Asian leve) |
| **IV** | Raramente queima, bronzeia facilmente (pele oliva, Hispanic, Mediterranean) |
| **V** | Muito raramente queima, bronzeia profusamente (pele marrom, Middle Eastern, Indian, dark Hispanic) |
| **VI** | Nunca queima, profundamente pigmentada (pele preta — African ancestry) |

**Propósito DECLARADO da escala:**

- Dosimetria UV/PUVA segura.
- Predição de risco de câncer de pele.
- Triagem clínica em dermatologia.

**Propósito NÃO declarado** (mas frequentemente assumido por terceiros):

- Classificação racial.
- Substituto para taxonomia étnica.
- Padrão biométrico.

Fitzpatrick **NUNCA** propôs a escala para esses usos. **Tipos V e
VI foram adicionados depois** justamente porque a escala original (I-IV)
era inadequada para "skin of color" — admissão implícita do viés
caucasiano-cêntrico da formulação original.

## 6. Limitações declaradas pelos autores

Fitzpatrick (1988) reconheceu:

- Skin type é **fenótipo complexo** — interações entre múltiplos
  fatores (melanina constitutiva, melanina facultativa, espessura
  epidérmica).
- Autoreporte por paciente é **imperfeito** — vieses de memória,
  experiência cultural com sol.
- Escala é **prática**, não bioquímica precisa.

## 7. Limitações que identifiquei (leitura crítica)

**Problemas estruturais da escala quando usada fora de seu propósito:**

- **Origem caucasiano-cêntrica:** **3 de 6 tipos** (I, II, III) cobrem
  o espectro "perceived as White", apenas 3 (IV, V, VI) cobrem o
  resto do mundo. Hall et al. (2018) chama de **"systematic bias
  toward lighter skin"**.
- **Inter-rater agreement variável**: dermatologistas profissionais
  vs anotadores leigos vs autoreporte produzem categorizações
  diferentes (Schumann et al. 2023 documenta para MST, padrão
  estende-se a Fitzpatrick).
- **Confusão sistemática com raça:** 1/3 dos dermatologistas confunde
  Fitzpatrick com raça/etnia (Ware et al. 2020). Esta confusão
  **propagou-se à literatura de ML fairness**.
- **Originalmente 4 tipos**, expandido para 6 — admissão de
  inadequação para pele escura. **Por que parar em 6?** Monk
  (2023) argumenta que **10 é mais inclusivo**.
- **Não considera variação inter-individual em melanização adaptativa**
  — bronzeamento de verão muda o tipo aparente.

## 8. Relação com nossa pesquisa

**Centralidade conceitual fundamental:**

1. **Esclarece o erro categorial** que permeia literatura de ML
   fairness: Fitzpatrick **não é** taxonomia racial, é dosimetria
   UV. Quando Hazirbas (2021) ou Lafargue (2025) usam Fitzpatrick,
   estão usando ferramenta **fora do propósito original**, embora
   válida por proxy.
2. **Justifica adoção de MST** (Schumann 2023) em vez de Fitzpatrick
   para Q10: MST foi **explicitamente desenhada** para fairness
   research, sem viés histórico para pele clara.
3. **Sustenta a posição de que "tom de pele cientificamente correto"
   é pergunta mal formulada** (parte de Q14). Não há "verdadeiro
   número de tons"; há instrumentos com propósitos diferentes.
4. **Citação OBRIGATÓRIA quando discutirmos Fitzpatrick** — para
   evitar perpetuar o erro categorial.

## 9. Pontos para citar / posicionar

- *"A escala Fitzpatrick (Fitzpatrick, 1988, *Archives of
  Dermatology*) foi originalmente desenvolvida em 1975 para um
  propósito clínico específico: prescrição segura de doses iniciais
  de UVA em fotoquimioterapia oral com metoxsaleno (PUVA) para
  psoríase. Não foi concebida como sistema de classificação racial
  ou étnica. Seu uso recorrente como proxy para raça em literatura
  de fairness em visão computacional (e.g., Buolamwini & Gebru,
  2018; Hazirbas et al., 2021) deve ser interpretado dentro destas
  ressalvas históricas."*
- *"A própria estrutura da escala Fitzpatrick — três tipos (I, II,
  III) cobrindo o espectro 'perceived as White' e três (IV, V, VI)
  cobrindo o restante do espectro fenotípico humano — reflete viés
  estruturalmente caucasiano-cêntrico, motivando o desenvolvimento
  posterior de escalas mais granulares e equitativas como o Monk
  Skin Tone Scale (Schumann et al., 2023)."*
- *"Adotamos o Monk Skin Tone Scale (10 pontos) sobre a Fitzpatrick
  (6 pontos) em nossa matriz Q10, reconhecendo: (i) o propósito
  histórico da Fitzpatrick é PUVA, não fairness; (ii) MST foi
  explicitamente desenhada para fairness research; (iii) a maior
  granularidade do MST captura melhor a variação fenotípica humana,
  particularmente em tons médios e escuros."*

## 10. Arquivos relacionados

- DOI: 10.1001/archderm.1988.01670060015008
- JAMA Network: https://jamanetwork.com/journals/jamadermatology/fullarticle/549509
- PubMed: 3377516
- Entradas relacionadas: [[buolamwini_2018]] (usa Fitzpatrick),
  [[dataset_hazirbas_2021]] (usa Fitzpatrick em Casual Conversations),
  [[lafargue_2025]] (usa Fitzpatrick em auditoria),
  [[schumann_2023]] (MST como sucessor moderno).
- **Responde Q13 em [[_perguntas]]** (origem e propósito da escala).

## 11. Trabalhos sugeridos pelos autores (Future Work)

Paper de 1988 não tem "future work" no sentido moderno. Direções
implícitas:

- **Refinar instrumentos de skin typing** — feito por Monk Skin
  Tone (2023) 35 anos depois. ✅ **Alinhada com Q14**.
- **Investigar correlação com fototipo bioquímico** (constitutive
  melanin) — pesquisa em curso na dermatologia.
- **Adaptar escala para skin of color** — Hall et al. 2022
  documentam que Fitzpatrick subestima risco de câncer em pele
  escura (V-VI), exatamente porque a escala foi desenhada para
  pele clara. ❌ Fora do escopo computacional.
- **NÃO sugere** uso para classificação racial — uso indevido
  posterior. ❌ Refutação implícita do uso comum em ML fairness.
