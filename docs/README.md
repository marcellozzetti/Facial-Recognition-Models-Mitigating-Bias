# Documentação — guia de navegação

> Reestruturado em **2026-05-25** após pivot estratégico decidido na
> reunião com o orientador (Prof. Marcos Quiles). A partir desta data,
> a documentação está dividida em **ativo** (material em uso e em
> construção) e **histórico** (material anterior preservado para
> rastreabilidade, mas que não deve ser citado diretamente em novos
> trabalhos sem revisão).

## Estrutura

```
docs/
├── README.md            ← este arquivo
├── ativo/               ← material vigente; ordem de leitura numerada 00→09
│   ├── 00_referencias.md       ← única fonte de verdade para citações
│   ├── 01_taxonomia.md         ← nomenclatura + glossário EN-PT
│   ├── 02_metodologia.md       ← protocolo experimental e métodos
│   ├── 03_metricas.md          ← métricas usadas, com procedência
│   ├── 04_survey/              ← uma ficha .md por paper VERIFIED e lido
│   ├── 05_landscape.md         ← síntese da paisagem da literatura
│   ├── 06_gap.md               ← identificação do gap real
│   ├── 07_thesis_statement.md  ← thesis statement v3
│   ├── 08_experimentos.md      ← experimentos reframados sobre o gap
│   └── 09_resultados.md        ← resultados consolidados
│
└── historico/           ← preservado; pode conter erros que foram corrigidos
    ├── README.md                ← explica origem e estado dos arquivos aqui
    ├── *.md                     ← docs anteriores à reestruturação
    ├── apresentações_arquivo/   ← apresentações antigas
    └── literature_corpus*.csv   ← corpus bibliográfico preservado
```

## Princípio de operação

1. **Toda citação** em qualquer documento de `ativo/` deve corresponder
   a uma entrada com status `✅ VERIFIED` em `ativo/00_referencias.md`.
   Sem verificação, sem citação.

2. **Toda síntese de conteúdo de um paper** deve ter um arquivo
   correspondente em `ativo/04_survey/<autor>_<ano>.md` baseado em
   leitura integral do PDF.

3. **Termos técnicos** seguem a taxonomia consolidada em
   `ativo/01_taxonomia.md`. Não introduzir termo novo sem registrá-lo lá.

4. **Reuso de material de `historico/`** requer re-verificação. Conteúdo
   pode ser citado apenas com indicação explícita da origem e da
   re-verificação realizada.

## O que NÃO está aqui

- Código-fonte (`src/`), configurações (`configs/`), scripts (`scripts/`)
  e saídas experimentais (`outputs/`) — preservados em sua organização
  original. Nenhum desses foi alterado pela reestruturação documental.
- Dependências (`requirements*.txt`, `pyproject.toml`).

## Status atual da reestruturação

- ✅ Diretórios `ativo/` e `historico/` criados.
- ✅ Citações erradas corrigidas em todos os documentos (movidos ou não).
  Cinco autorias estavam confabuladas; verificação documentada em
  `ativo/00_referencias.md`.
- ⏳ `ativo/01_taxonomia.md` em construção.
- ⏳ `ativo/02_metodologia.md`, `03_metricas.md`, `04_survey/` —
  pendentes (parte do plano de trabalho aprovado em 2026-05-25).
- ⏳ Survey rigoroso da literatura de fairness em classificação racial
  facial — pendente.
