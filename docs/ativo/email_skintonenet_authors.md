# Rascunho de email — solicitação de acesso antecipado ao SkinToneNet

**Contexto para o Marcello:**
Este é um rascunho para envio pelo **orientador (Prof. Marcos Quiles)** aos autores
do SkinToneNet (Matias, Costa, Neto & Novello de Brito, 2026 —
[arXiv:2603.02475](https://arxiv.org/abs/2603.02475)). Envio institucional
(orientador → orientadores) tem taxa de resposta muito maior que cold
email discente. Ajustar o texto conforme preferência do Prof. Quiles
antes do envio.

**Destinatários sugeridos:**
- Prof. **João Batista Neto** (ICMC/USP) — provável orientador do trabalho
- Prof. **Tiago Novello de Brito** (IMPA) — coautor sênior
- CC: Vitor Pereira Matias (primeiro autor) e Márcus Vinícius Lobo Costa

**Endereços a confirmar:** procurar nos sites institucionais
(icmc.usp.br, impa.br) — não os incluí aqui para não vazar dados errados.

---

## Assunto

Solicitação de acesso antecipado — pesos SkinToneNet (arXiv:2603.02475)
para pesquisa de mestrado sobre mitigação de viés racial em RF

---

## Corpo do email

Prezado Prof. João Batista Neto,
Prezado Prof. Tiago Novello de Brito,
(cc: Vitor Pereira Matias, Márcus Vinícius Lobo Costa)

Escrevo em nome do meu orientando de mestrado, **Marcello Vinicius Alves
Ozzetti Cruz**, aluno regular do Programa de Pós-Graduação em Ciência da
Computação da UNIFESP/ICT. Sua dissertação, com qualificação agendada
para 30 de setembro de 2026, propõe um pipeline de mitigação de viés
racial em classificação facial condicionado ao tom de pele (Monk Skin
Tone), tendo o **SkinToneNet como classificador MST de referência** na
etapa inicial do método.

O trabalho de vocês — *"Large-Scale Dataset and Benchmark for Skin Tone
Classification in the Wild"* (arXiv:2603.02475) — é a base metodológica
central do Capítulo 4 da dissertação e a referência declarada no Objetivo
2 (validação estratificada por tom de pele). No documento consta que
"code and data will be available soon", e temos acompanhado a página do
paper aguardando a divulgação pública.

**Nosso pedido:** seria possível obter acesso antecipado (i) aos pesos
pré-treinados do SkinToneNet (ViT-Small fine-tuned em STW) e (ii),
idealmente, ao dataset STW, exclusivamente para uso acadêmico no escopo
da dissertação do Marcello?

Comprometemo-nos formalmente com:

1. **Citação correta** em todos os artefatos derivados (dissertação,
   eventuais publicações, código publicado no GitHub);
2. **Não redistribuição** dos pesos ou do dataset — uso restrito ao
   grupo de pesquisa (Marcello + orientador);
3. **Aderência à licença** que vocês eventualmente definirem
   (CC BY 4.0, MIT ou outra);
4. **Feedback experimental**: podemos compartilhar métricas de
   generalização do SkinToneNet sobre o subconjunto de validação
   FairFace (~10.954 imagens, 7 grupos raciais) que vamos utilizar,
   caso seja de interesse para o benchmark de vocês.

Para contexto adicional, o pipeline do Marcello envolve **6 etapas
metodológicas** (Cap. 4 §4.2 da dissertação): (1) classificador MST via
SkinToneNet, (2) auditoria fenotípica FairFace × MST, (3) classificador
racial condicionado por FiLM alimentado pela saída softmax do
SkinToneNet, (4) comparação contra 6 baselines de *fairness*
(ResNet-34, ConvNeXt-T puro, FSCL+, Group DRO, FineFACE, Adversarial
Debiasing), (5) transferência para verificação (RFW/BFW), e (6)
decomposição de erros. O SkinToneNet é o **insumo crítico da Etapa 1**;
sem os pesos, precisaríamos reproduzir o treinamento a partir do
STW quando este estiver disponível — o que atrasaria o cronograma em
pelo menos dois meses.

Se preferirem canal formal — carta institucional assinada pela
coordenação do PPG-CC UNIFESP/ICT ou via nossa área de convênios
acadêmicos — podemos providenciar. Estamos abertos, também, a discutir
qualquer forma de colaboração ou reconhecimento que julguem apropriada.

Ficamos à disposição para esclarecimentos e agradecemos, desde já, pela
atenção.

Cordialmente,

**Prof. Dr. Marcos Gonçalves Quiles**
Programa de Pós-Graduação em Ciência da Computação
Instituto de Ciência e Tecnologia — UNIFESP
São José dos Campos — SP

_Orientando: Marcello Vinicius Alves Ozzetti Cruz_
_Repositório de trabalho: [placeholder — decidir se compartilhar]_

---

## Checklist antes do envio

- [ ] Confirmar emails institucionais dos destinatários (icmc.usp.br / impa.br)
- [ ] Revisar com Prof. Quiles (formatação e tom)
- [ ] Decidir se anexa o Cap. 4 da dissertação (proposta metodológica formal)
- [ ] Decidir se compartilha URL do repositório GitHub ou mantém privado
- [ ] Assinatura institucional formatada do Prof. Quiles
- [ ] Prazo mental de resposta: 2 semanas; se sem retorno até 2026-09-01, reenvio cortês
