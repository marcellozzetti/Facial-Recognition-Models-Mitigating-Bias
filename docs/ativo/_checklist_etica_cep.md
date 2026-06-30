---
data: 2026-06-30
tipo: checklist-administrativo
escopo: adequacao etica do projeto a Resolucao 200/2021/CONSU Unifesp
status: ativo
proxima_acao: baixar modelo Declaracao de Responsabilidade
---

# Checklist — Adequação ética do projeto (CEP/Unifesp)

> **Base legal**: Resolução nº 200/2021/CONSELHO UNIVERSITÁRIO
> (Unifesp), SEI 0719529.
> **Enquadramento do projeto**: **Art. 8º** — pesquisa que não
> envolve seres humanos direta ou indiretamente, **dispensada de
> cadastro no CEP**, exigindo apenas **Declaração de
> Responsabilidade**.

## 1. Justificativa do enquadramento

Após adequação da v3.6 dos objetivos da tese, o projeto:

| Critério | Status | Justificativa |
|---|---|---|
| Coleta primária de dados de seres humanos | ❌ Não | Usa apenas datasets públicos secundários (FairFace, RFW, BFW, BUPT-Balancedface) |
| Intervenção direta em seres humanos | ❌ Não | Pesquisa puramente computacional |
| Uso de animais vertebrados vivos | ❌ Não | Não se aplica |
| Crowdsourcing externo para anotação (Prolific etc.) | ❌ Não — **removido na v3.6** | Validação MST agora é interna (Mestrando + Orientador, ~200-300 imgs) |
| Datasets contêm faces humanas | ✅ Sim, mas | Datasets já anotados e disponibilizados por terceiros sob licença de pesquisa; sem identificação pessoal coletada pelo projeto |

**Conclusão**: projeto **se enquadra no Art. 8º** — pesquisa que
não envolve seres humanos direta ou indiretamente.

## 2. Documento exigido: Declaração de Responsabilidade

### 2.1 Onde obter o modelo

📌 **Site oficial do CEP/Unifesp**: http://www.cep.unifesp.br/cep

Procurar por "Declaração de Responsabilidade" ou "Formulários" na
página do CEP.

### 2.2 Quem assina (3 assinaturas obrigatórias)

| # | Quem | Responsabilidade |
|---|---|---|
| 1 | **Estudante** — Marcello Ozzetti (Mestrando) | Confirma que conduzirá a pesquisa em conformidade com a Resolução 200/2021 |
| 2 | **Orientador / Pesquisador Responsável** — Prof. Marcos Quiles | Atesta que o projeto não envolve direta ou indiretamente seres humanos nem animais vertebrados |
| 3 | **Chefe do Departamento** ao qual o orientador está vinculado | A confirmar com o orientador — provavelmente Departamento de Ciência da Computação (ICT/Unifesp) ou equivalente |

### 2.3 Para onde enviar / anexar

- **Anexar ao projeto de pesquisa** (qualificação)
- **Apresentar ao Programa de Pós-Graduação** em Ciência da
  Computação da Unifesp/ICT
- Manter cópia assinada no repositório local (não commitada — PII
  do chefe de departamento)

## 3. Sequência de ações

### Passo 1 — Confirmar chefe do departamento

- [ ] **Marcello**: perguntar ao Prof. Quiles na próxima reunião
  quem é o(a) chefe do Departamento que assinará.

### Passo 2 — Baixar modelo

- [ ] Acessar http://www.cep.unifesp.br/cep
- [ ] Localizar e baixar o PDF/DOC do modelo de Declaração de
  Responsabilidade
- [ ] Salvar localmente (fora do repositório versionado se contiver
  template pessoal)

### Passo 3 — Preencher

Campos típicos do modelo (estimativa baseada na Resolução):

| Campo | Valor a preencher |
|---|---|
| Título do projeto | "Equidade Racial em Classificação Facial: Pipeline Condicionado por Tom de Pele (MST) via FiLM-Conditioning sobre FairFace" (ou versão sintética acordada com orientador) |
| Programa de Pós-Graduação | Mestrado em Ciência da Computação — Unifesp/ICT |
| Estudante | Marcello Ozzetti |
| Orientador | Prof. Marcos Quiles |
| Justificativa de dispensa CEP | "Pesquisa puramente computacional sobre datasets secundários publicamente disponíveis (FairFace, RFW, BFW). Não envolve coleta primária de dados de seres humanos nem crowdsourcing externo. Validação manual em subset realizada por equipe acadêmica interna." |

### Passo 4 — Coletar assinaturas

- [ ] Estudante assina
- [ ] Solicitar assinatura do Orientador
- [ ] Solicitar assinatura do Chefe de Departamento
- Sugestão: usar **SEI Unifesp** para tramitação eletrônica (mais
  rápido que assinaturas físicas)

### Passo 5 — Anexar à qualificação

- [ ] Incluir a Declaração assinada como **Anexo** da qualificação
- [ ] Mencionar explicitamente no Capítulo 4 (Metodologia) que o
  projeto se enquadra no Art. 8º da Res. 200/2021/CONSU

## 4. Modelo de parágrafo metodológico (para Cap 4)

> Sugestão de texto a incluir no Capítulo 4 da qualificação:

> *"Quanto aos aspectos éticos da pesquisa, este projeto se enquadra
> no Artigo 8º da Resolução nº 200/2021 do Conselho Universitário
> da Unifesp, por consistir em pesquisa puramente computacional
> sobre datasets secundários publicamente disponíveis (FairFace,
> RFW, BFW), sem coleta primária ou tratamento de dados de seres
> humanos nem envolvimento indireto via plataformas de crowdsourcing
> externo. A validação manual prevista no Objetivo Específico 1
> sobre subset estratificado do FairFace será conduzida internamente
> pela equipe acadêmica do projeto (Mestrando e Orientador), o que
> dispensa submissão ao Comitê de Ética em Pesquisa (CEP) da Unifesp.
> A Declaração de Responsabilidade prevista no parágrafo único do
> referido Artigo 8º, assinada por estudante, orientador e chefe
> do Departamento, encontra-se anexa a este documento."*

## 5. Caso o cenário mude

Se em algum momento futuro a pesquisa **precisar** envolver
crowdsourcing externo ou coleta primária de dados de seres humanos
(por exemplo, retomar Prolific após adequações ou conduzir survey):

1. **Suspender a atividade** imediatamente.
2. **Submeter projeto ao CEP** ANTES de iniciar (Art. 1º — vedado
   submeter projetos já iniciados, conforme parágrafo único).
3. **Aguardar aprovação** antes de retomar.
4. **Atualizar este checklist** registrando a mudança.

## 6. Próximos passos consolidados

| Prazo | Ação | Responsável |
|---|---|---|
| Próxima reunião | Confirmar chefe de departamento com orientador | Marcello |
| Esta semana | Baixar modelo no site do CEP/Unifesp | Marcello |
| Esta semana | Preencher e iniciar coleta de assinaturas via SEI | Marcello |
| Antes da entrega (15/jul ou prorrogado) | Declaração assinada anexada à qualificação | Marcello |
| Cap 4 da qualificação | Incluir parágrafo metodológico sobre dispensa CEP | Marcello na escrita |

## Referências

- **Resolução 200/2021/CONSU** — Unifesp (SEI 0719529).
  Documento-base do enquadramento.
- **Art. 8º** — dispensa de cadastro CEP para pesquisas sem
  envolvimento de seres humanos.
- **Parágrafo único do Art. 8º** — exigência da Declaração de
  Responsabilidade.
- **Site CEP/Unifesp**: http://www.cep.unifesp.br/cep
