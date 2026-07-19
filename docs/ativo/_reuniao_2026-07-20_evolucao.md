---
data: 2026-07-20
tipo: material-apresentacao
participantes: [Marcello Ozzetti, Prof. Marcos Quiles]
reuniao_anterior_realizada: 2026-07-13
atualizacao_unica: fechamento da escrita da qualificacao
status: preparacao
---

# Reunião com o orientador — 20/jul/2026

> **Atualização única desta semana**: a escrita da qualificação está
> fechada. Cinco capítulos completos, migrados para o template
> oficial abnTeX2 da Unifesp, com padrão de citação ABNT NBR 10520,
> 42 referências validadas e glossário anexo com 73 termos.

---

## 1. Plano até a defesa — sem alteração

| # | Marco | Data | Status |
|---|---|---|---|
| 1 | Primeira revisão da qualificação ao orientador | 15/jul/2026 | Entregue (Overleaf compartilhado desde 13/jul) |
| 2 | Pedido formal de qualificação ao PPG-CC / ICT | 30/jul/2026 | **Em curso — 10 dias** |
| 3 | Defesa da qualificação | outubro/2026 | Sujeita à prorrogação de 2 meses solicitada |

---

## 2. Atualização única — escrita da qualificação fechada

### 2.1 O que fechou nesta semana

Um único bloco de trabalho concentrou o esforço da semana: a
finalização estrutural do documento da qualificação. Não há novidade
metodológica nem científica; a novidade é operacional --- o texto
agora está em forma acadêmica finalizada, pronto para o pedido
formal de qualificação.

Cinco entregas específicas concluídas:

1. **Cap 2 (Revisão) finalizado.** O capítulo mais denso, que estava
   em revisão ativa na semana passada, foi consolidado com as 104
   fichas mapeadas nas seis frentes teóricas, incluindo a nova
   subseção sobre alternativas de conditioning ponderadas (FiLM vs
   CBN vs cross-attention vs AdaIN vs SPADE vs HyperNet vs LoRA vs
   Adapter) --- resposta antecipada a possível questionamento de
   banca sobre a escolha arquitetural.

2. **Migração para o template oficial abnTeX2 da Unifesp/ICT.**
   Anteriormente o Overleaf usava esqueleto próprio; agora está
   sobre o template institucional (classe `icmc.cls`, estilo
   bibliográfico `abntexalfenglish.bst`) que o senhor tinha
   compartilhado, com título em PT e EN, dados do PPGCC, opção
   `qualificacao` e todos os elementos pré-textuais no formato
   correto.

3. **Adequação ao padrão de citação ABNT NBR 10520.** Migração dos
   comandos de `biblatex` (`\autocite`, `\textcite`) para o padrão
   BibTeX clássico da abnTeX2:
   - `\cite{X}` --- citação indireta, renderiza "(AUTOR, ano)"
   - `\citeonline{X}` --- citação nominal, renderiza "Autor (ano)"
   - `\citeauthoronline{X}` --- só o autor, renderiza "Autor"

4. **Glossário anexo com 73 termos** organizados em nove seções
   temáticas: datasets faciais e de tom de pele, modelos e
   arquiteturas, escalas de tom de pele, mecanismos de conditioning,
   métricas de fairness e estatística, baselines de mitigação,
   conceitos técnicos gerais, conceitos de fairness / sociologia /
   legal, instituições e normas regulatórias.

5. **Validação sistemática das 42 chaves BibTeX** contra o arquivo
   `references.bib` (104 entradas). Zero chaves ausentes, zero
   comandos remanescentes de `biblatex`, zero labels inconsistentes.

### 2.2 Estrutura fechada dos elementos textuais

Seguindo a norma ABNT NBR 14724, os elementos textuais organizam-se
em cinco capítulos:

| Capítulo | Palavras | Foco |
|---|---:|---|
| 1. Introdução | ~1200 | Contexto (NIST/FRVT), problema (F1 60% Latinx), diagnóstico intra-Latinx, implicações regulatórias (European AI Act) |
| 2. Revisão de Literatura | ~1985 | Seis frentes teóricas + tripé Latinx (Telles / Bryc / Pew) + alternativas de conditioning ponderadas + cinco lacunas |
| 3. Objetivos, Hipóteses e Contribuições | ~896 | Objetivo geral + 6 OEs + 6 hipóteses testáveis + 7 contribuições esperadas |
| 4. Metodologia | ~1200 | Pipeline em 6 etapas + 4 configurações comparativas + ética CEP formal (Art.~8º) |
| 5. Cronograma e Próximos Passos | ~692 | Marcos + riscos + próximos passos + considerações finais |
| **Total** | **~5973** | (fora resumo e glossário) |

O documento é complementado por:

- **Resumo** de 500 palavras sem citações (conforme NBR 6028)
- **Glossário** com 73 termos (elemento pós-textual)
- **Referências bibliográficas** com 104 entradas verificadas

### 2.3 Organização de arquivos no repositório

```
docs/tese/
├── thesis.tex                      arquivo mestre
├── references.bib                  bibliografia (104 entradas)
├── packages/                       classe icmc + estilo BST + capa
├── images/                         logo Unifesp
└── tex/
    ├── pre-textual/                dedicatória, agradecimentos,
    │                               epígrafe, ficha catalográfica
    ├── textual/                    os 5 capítulos
    │   ├── introducao.tex
    │   ├── revisao-literatura.tex
    │   ├── objetivos.tex
    │   ├── metodologia.tex
    │   └── plano-trabalho.tex
    ├── newword.tex                 setup do glossário
    └── glossario.tex               73 termos técnicos
```

---

## 3. Solicitação pessoal --- extensão de dois meses

Reiteração da solicitação já apresentada na reunião anterior:

- **Motivo**: nascimento de minha filha em 28/março/2026 e
  afastamento subsequente para cuidado parental.
- **Marcos preservados**: primeira revisão (15/jul --- feita) e
  pedido formal (30/jul --- próximos 10 dias).
- **Solicitação**: prorrogação de dois meses no prazo da defesa,
  de agosto para outubro de 2026.
- **Documento**: carta redigida em dois parágrafos, aguardando
  orientação sobre trâmite formal (carta direta ao orientador,
  requerimento via SEI ou processo com anexos).

---

## 4. Perguntas objetivas ao orientador

1. **Overleaf**: o senhor teve chance de percorrer algum capítulo?
   Algum ponto crítico para ajustar antes do pedido formal de 30/jul?
2. **Ficha catalográfica**: pela Biblioteca da Unifesp, qual o
   prazo típico? Precisa ser solicitada antes de 30/jul ou pode
   entrar na versão pós-defesa?
3. **Trâmite da extensão de 2 meses**: alguma decisão sobre o
   caminho formal?
4. **Co-orientador**: alguma definição para constar na versão
   formal do pedido?
5. **Banca preliminar**: sugestões de nomes para começarmos a
   articular contatos?

---

## 5. Próximos passos --- semana 21-30/jul

1. Aplicar os ajustes que sair desta reunião ao Overleaf.
2. Confirmar procedimento com a secretaria do PPG-CC quanto a
   documentos necessários para o pedido formal.
3. Protocolar carta de solicitação de extensão de 2 meses.
4. Submeter o **pedido formal de qualificação** em 30/jul.
5. Iniciar preparação técnica do Cap.~1 experimental (auditoria
   MST × raça), primeiro bloco experimental pós-qualificação.

---

## Anexos

- [_reuniao_2026-07-20_cola.md](_reuniao_2026-07-20_cola.md) --- cola de bolso
- [_objetivo_tese_v3.3.md](_objetivo_tese_v3.3.md) --- v3.6
- [_pre_qualificacao_narrativa.md](_pre_qualificacao_narrativa.md) --- v1.2
- `docs/tese/thesis.tex` --- documento mestre da qualificação
- `docs/tese/references.bib` --- bibliografia consolidada
