# Tese — Qualificação de Mestrado (Marcello Ozzetti / Unifesp-ICT)

Estrutura oficial da dissertação em **LaTeX/abnTeX2**, seguindo o
template ICMC adaptado ao padrão Unifesp-ICT.

## Arquivo principal

- **`thesis.tex`** — arquivo a compilar (comando `pdflatex + bibtex + pdflatex + pdflatex`, ou `latexmk -pdf` no Overleaf)

## Estrutura de diretórios

```
docs/tese/
├── thesis.tex                    ← arquivo mestre
├── references.bib                ← bibliografia (104 entradas verificadas)
├── packages/
│   ├── icmc.cls                  ← classe LaTeX (abnTeX2 + adaptações ICMC)
│   ├── abntexalfenglish.bst      ← estilo bibliográfico ABNT
│   ├── capa.pdf                  ← capa institucional
│   ├── contra-capa.png
│   └── fonts/
├── images/
│   └── Unifesp_completa_policromia_RGB.png
└── tex/
    ├── pre-textual/                     ← Elementos pré-textuais
    │   ├── dedicatoria.tex
    │   ├── agradecimentos.tex
    │   ├── epigrafe.tex
    │   └── ficha-catalografica.pdf
    ├── textual/                         ← Elementos textuais (5 capítulos)
    │   ├── introducao.tex               ← Capítulo 1
    │   ├── revisao-literatura.tex       ← Capítulo 2
    │   ├── objetivos.tex                ← Capítulo 3
    │   ├── metodologia.tex              ← Capítulo 4
    │   └── plano-trabalho.tex           ← Capítulo 5 (Cronograma)
    ├── newword.tex                      ← comando \newword + ambiente {glossario}
    └── glossario.tex                    ← 73 termos técnicos (pós-textual)
```

## Padrão de citação

Sistema **autor-data ABNT** (NBR 10520) via BibTeX clássico + `abntexalfenglish.bst`:

| Comando | Renderização | Uso |
|---|---|---|
| `\cite{X}` | (KÄRKKÄINEN, 2021) | Citação implícita |
| `\citeonline{X}` | Kärkkäinen (2021) | Citação nominal (autor no texto) |
| `\citeauthoronline{X}` | Kärkkäinen | Só o autor, sem ano |
| `\citeyear{X}` | 2021 | Só o ano |

**Não usar** `\autocite` nem `\textcite` (são do `biblatex`, incompatíveis com esta classe).

## Como compilar

### No Overleaf
1. Fazer upload da pasta `docs/tese/` inteira
2. Configurar `thesis.tex` como arquivo principal
3. TeX Live 2021+ (padrão do Overleaf)
4. Compilador: **pdfLaTeX** (não LuaLaTeX nem XeLaTeX)

### Localmente (MiKTeX / TeXLive)
```bash
cd docs/tese
pdflatex thesis
bibtex thesis
pdflatex thesis
pdflatex thesis
```

Ou:
```bash
latexmk -pdf thesis
```

## Histórico

- **`docs/historico/tese_biblatex_backup/`** — versão anterior em biblatex (antes da migração para o template abnTeX2/ICMC). Preservada para consulta.
