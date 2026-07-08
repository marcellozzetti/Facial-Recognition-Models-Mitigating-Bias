---
data: 2026-07-08
tipo: material-apresentacao
participantes: [Marcello Ozzetti, Prof. Marcos Quiles]
reuniao_anterior_realizada: 2026-06-15
gap: 3 semanas sem reuniões
proxima_entrega: 2026-07-15
status: preparacao
---

# Reunião com o orientador — 08/jul/2026

> **Consolidação após 3 semanas** sem reuniões. Foco em plano de
> trabalho ajustado, estado da escrita e duas pendências específicas
> que precisam de alinhamento hoje.

---

## 1. Plano de trabalho consolidado

Três marcos formais no horizonte próximo:

| # | Marco | Data | Ajuste em relação ao original |
|---|---|---|---|
| 1 | **1ª revisão da qualificação ao orientador** | 15/jul/2026 | Mantido |
| 2 | **Pedido formal de qualificação (PPG-CC / ICT)** | 30/jul/2026 | Mantido |
| 3 | **Defesa da qualificação** | **outubro/2026** | **De agosto para outubro** (por solicitação de extensão de 2 meses) |

**Racional dos marcos:**

- Marco 1 (15/jul): mantido para permitir uma primeira rodada de
  feedback do orientador, mesmo que ainda haja ajustes.
- Marco 2 (30/jul): mantido pelo prazo regimental do PPG.
- Marco 3 (outubro): ajuste solicitado por conta do afastamento
  parental (28/mar/2026 e semanas subsequentes).

---

## 2. Pendência 1 — Escrita da qualificação

### 2.1 Estado atual do corpo da tese

Toda a base narrativa está preparada e versionada no repositório:

| Capítulo | Base no repositório | Estado |
|---|---|---|
| 1. Introdução | `_pre_qualificacao_narrativa.md` v1.2 | Corpo pronto |
| 2. Revisão bibliográfica | 104 fichas + `_mapa_citacoes_por_capitulo.md` | Corpo pronto |
| 3. Objetivos | `_objetivo_tese` v3.6 (OG + 6 OE + 6 H + 7 C) | Corpo pronto |
| 4. Metodologia | `_decisao_arquitetural_film.md` + `_validacao_cientifica_pipeline.md` + `_checklist_etica_cep.md` | Corpo pronto |
| 5. Cronograma | Marcos ajustados desta reunião | A finalizar após reunião |

### 2.2 Metodologia de trabalho — postura sobre uso de LLM

Ponto crítico para deixar claro ao orientador:

> **Este trabalho NÃO é 100% LLM.** A LLM (Claude Code) foi usada
> como **ferramenta de apoio operacional**, não como autora
> substituta.

**Divisão explícita de responsabilidade:**

| Atividade | Autor | Papel da LLM |
|---|---|---|
| Definição do problema e da pergunta de pesquisa | Marcello | Nenhum |
| Escolha do dataset FairFace + escala MST | Marcello (via reuniões) | Nenhum |
| Leitura crítica dos 104 papers | Marcello (VPN, downloads, PDFs) | Assistência em resumos após leitura |
| Ficha bibliográfica de cada paper | Marcello (verificação PDF a PDF) | Formatação da estrutura padronizada |
| Argumentação e storytelling da tese | Marcello (decisões editoriais) | Sugestão de estrutura + revisão de coerência |
| Decisão arquitetural FiLM | Marcello + Orientador (reuniões) | Comparação sistemática entre alternativas |
| Validação NotebookLM | Marcello (ferramenta externa Google) | Nenhum (segunda opinião independente) |
| Ajuste ético do OE-1 (Prolific → interno) | Marcello (interpretação Res. 200/2021) | Análise da resolução |
| Transferência para Overleaf | **Marcello — manual, linha a linha** | Nenhum |
| Revisão final de cada capítulo | Marcello | Revisão auxiliar de português |

**Princípio adotado**: cada linha do texto final é escrita, lida e
apropriada conscientemente pelo mestrando. A LLM é ferramenta
operacional — como IDE, corretor ortográfico e mecanismo de busca —
e não substitui a construção intelectual do argumento.

### 2.3 Estratégia de transferência para Overleaf

- **Modalidade**: transferência **manual**, capítulo por capítulo,
  parágrafo por parágrafo.
- **Vantagem 1**: consistência de formato e citações ABNT.
- **Vantagem 2**: revisão obrigatória de cada trecho no momento
  da transferência.
- **Vantagem 3**: apropriação consciente e defensável do texto.
- **Cronograma**: 08-14/jul para conversão + revisão + primeira
  entrega em 15/jul.

### 2.4 Se o orientador quiser ver

O corpo dos capítulos está no repositório GitHub e pode ser aberto
ao vivo durante a reunião:

```
docs/tese/capitulos/
  ├── 01_introducao.tex       (esqueleto — corpo em ativo/)
  ├── 02_revisao_literatura.tex
  ├── 03_objetivos.tex
  ├── 04_metodologia.tex
  └── 05_cronograma.tex

docs/ativo/                    (corpo narrativo preparado)
  ├── _pre_qualificacao_narrativa.md   (base do Cap 1)
  ├── _objetivo_tese_v3.3.md           (base do Cap 3)
  ├── _decisao_arquitetural_film.md    (base do Cap 4)
  └── _mapa_citacoes_por_capitulo.md   (mapa das 104 fichas)
```

---

## 3. Pendência 2 — Adequação ética CEP/Unifesp

### 3.1 Análise normativa

Resolução avaliada: **Nº 200/2021/CONSELHO UNIVERSITÁRIO Unifesp**
(SEI 0719529).

Documento completo com auditoria em [_checklist_etica_cep.md](_checklist_etica_cep.md).

### 3.2 Enquadramento decidido: Art. 8º

| Critério | Avaliação |
|---|---|
| Coleta primária de dados de seres humanos | Não |
| Intervenção direta em seres humanos | Não |
| Uso de animais vertebrados vivos | Não |
| Crowdsourcing externo (Prolific etc.) | **Removido na v3.6** |
| Uso de datasets secundários públicos | Sim (FairFace, RFW, BFW, BUPT) |

**Conclusão**: projeto enquadra-se no **Art. 8º** — dispensa de
cadastro no CEP; exige apenas **Declaração de Responsabilidade**.

### 3.3 Ajuste técnico no OE-1 (v3.5 → v3.6)

| Aspecto | Versão v3.5 (anterior) | Versão v3.6 (atual) |
|---|---|---|
| Método de validação MST | Crowdsourcing externo via Prolific | Validação interna pela equipe acadêmica |
| Anotadores | ~3 externos pagos | Mestrando + Orientador |
| Escala | ~700 imgs estratificadas | ~200-300 imgs estratificadas |
| Estratificação | Por raça e tom MST | Por raça e tom MST (**preservada**) |
| Implicação ética | Pesquisa indireta com humanos → requer CEP | Dispensa CEP → apenas Declaração |

### 3.4 Trâmite administrativo — próximos passos

1. **Confirmar hoje** com o orientador quem é o(a) Chefe do
   Departamento (3ª assinatura obrigatória).
2. **Baixar modelo** de Declaração em http://www.cep.unifesp.br/cep.
3. **Preencher** com dados do projeto.
4. **Coletar 3 assinaturas** via SEI Unifesp: Mestrando +
   Orientador + Chefe do Departamento.
5. **Anexar Declaração assinada** à entrega da qualificação.
6. **Incluir parágrafo metodológico** no Cap 4 sobre dispensa CEP
   (Art. 8º Res. 200/2021).

---

## 4. Solicitação pessoal — extensão de 2 meses

- **Motivo**: nascimento de minha filha em **28/março/2026** e
  período de afastamento subsequente para cuidado parental.
- **Impacto no cronograma**: prazo da defesa da qualificação
  passa de **agosto → outubro/2026**.
- **Marcos preservados**: entrega da 1ª revisão (15/jul) e pedido
  formal (30/jul) **mantidos** — a extensão afeta apenas o prazo
  final da defesa.
- **Documento**: carta formal em 2 parágrafos redigida e pronta
  para envio via SEI.
- **Pergunta ao orientador**: qual é o trâmite formal correto
  (carta direta, requerimento via SEI ao PPG, processo com anexos)?

---

## 5. Perguntas objetivas ao orientador

1. **Estrutura dos capítulos** apresentada está adequada?
   Alguma sugestão de reorganização?
2. **Chefe do Departamento** — quem deve assinar a Declaração
   de Responsabilidade do CEP?
3. **Trâmite formal da extensão de 2 meses** — carta, requerimento
   SEI, processo com anexos?
4. Alguma orientação sobre **template Overleaf institucional** ou
   padrão ABNT específico Unifesp/ICT?
5. **Co-orientador** — alguma definição?
6. Alguma sugestão sobre **composição da banca preliminar**?

---

## 6. Próximos passos imediatos (semana 08-14/jul)

1. Transferir corpo do Cap 1 do Markdown para o Overleaf (dia 08-09).
2. Transferir Cap 2 (dia 10-11).
3. Transferir Caps 3-5 (dia 12-13).
4. Revisão final integrada (dia 14).
5. **Entrega da 1ª revisão da qualificação em 15/jul.**
6. Iniciar tramitação da Declaração de Responsabilidade em paralelo.
7. Protocolar carta de solicitação de extensão em paralelo.

---

## Anexos

- [_objetivo_tese_v3.3.md](_objetivo_tese_v3.3.md) — v3.6 com adequação ética
- [_pre_qualificacao_narrativa.md](_pre_qualificacao_narrativa.md) — base do Cap 1
- [_mapa_citacoes_por_capitulo.md](_mapa_citacoes_por_capitulo.md) — 104 fichas alocadas
- [_decisao_arquitetural_film.md](_decisao_arquitetural_film.md) — base do Cap 4
- [_checklist_etica_cep.md](_checklist_etica_cep.md) — enquadramento Art. 8º
- [_validacao_cross_reference_v3.md](_validacao_cross_reference_v3.md) — auditoria tese × corpus
- `docs/tese/referencias.bib` — bibliografia consolidada (104 entradas)
