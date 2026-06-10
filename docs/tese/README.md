# Estrutura LaTeX da qualificação

Arquivos LaTeX prontos para integração no Overleaf (template Unifesp/ICT).

## Estrutura

```
docs/tese/
├── README.md                        (este arquivo)
├── main.tex                         (esqueleto principal — adaptar ao template)
├── capitulos/
│   ├── 00_resumo.tex                (resumo + abstract)
│   ├── 01_introducao.tex            (Contexto + Problemas)
│   ├── 02_revisao_literatura.tex    (Estado-da-arte + Gap)
│   ├── 03_objetivos.tex             (Objetivos + Hipóteses + Contribuições)
│   ├── 04_metodologia.tex           (Pipeline 6 etapas + Protocolo)
│   └── 05_cronograma.tex            (Cronograma + Próximos passos)
└── referencias.bib                  (Bibliografia BibTeX)
```

## Como usar no Overleaf

1. Abrir seu projeto Overleaf existente (template Unifesp/ICT).
2. Fazer upload dos arquivos `.tex` da pasta `capitulos/`.
3. Fazer upload de `referencias.bib`.
4. No arquivo principal do Overleaf, importar capítulos via:
   ```latex
   \input{capitulos/01_introducao}
   \input{capitulos/02_revisao_literatura}
   \input{capitulos/03_objetivos}
   \input{capitulos/04_metodologia}
   \input{capitulos/05_cronograma}
   ```
5. Adicionar `\bibliography{referencias}` antes do `\end{document}`.

## Fontes de conteúdo

- **Texto narrativo**: extraído de [_pre_qualificacao_narrativa.md](../ativo/_pre_qualificacao_narrativa.md)
- **Objetivos + hipóteses**: [_objetivo_tese_v3.3.md](../ativo/_objetivo_tese_v3.3.md)
- **Referências**: 55 fichas do [corpus](../ativo/04_pesquisa_bibliografica/INDEX.md)

## Próximos passos

1. Adaptar `main.tex` ao template Unifesp (mudar documentclass, etc).
2. Refinar cada capítulo com revisão crítica.
3. Adicionar figuras (FiLM diagram, ConvNeXt-T, pipeline).
4. Verificar referências BibTeX vs corpus.
5. Polir linguagem acadêmica formal.

## Deadline

**Submissão**: 15 de julho de 2026 (~35 dias a partir de 2026-06-10).
