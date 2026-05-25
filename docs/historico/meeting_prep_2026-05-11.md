# Preparação para a Reunião com o Coordenador — 2026-05-11 (segunda-feira)

**Programa-alvo:** Ciência da Computação — Unifesp / ICT (São José dos Campos)
**Objetivo da reunião:** apresentar a proposta de mestrado e demonstrar prontidão e domínio técnico do tema.
**Duração estimada:** 1 dia produtivo (~8h, divididos em 4 blocos).

## Princípio do plano

Você não precisa virar especialista no SOTA inteiro em um dia — precisa **dominar o seu próprio argumento** e ter **mapeado as 5–6 perguntas mais prováveis** do coordenador.

A reunião provavelmente vai cobrir três coisas, nesta ordem:
1. **"Qual é a pergunta de pesquisa, e por que ela é interessante?"** (~30% do tempo)
2. **"Como o que você já fez justifica a pergunta?"** (~40%) — aqui você manda muito bem porque fez
3. **"Como você vai responder, em que tempo, com que recursos?"** (~30%)

Foque a leitura no que sustenta esses três blocos — não tente ler 50 papers.

---

## Estrutura do dia

### Bloco 1 (manhã, ~3h) — As 4 leituras essenciais

Estas são as **referências de fundo** que sustentam todos os argumentos da proposta. Ler com atenção total, fazer fichamento curto (1 parágrafo cada).

#### 1.1 Kotwal & Marcel (2025) — *Review of Demographic Fairness in Face Recognition* (~60 min)
**Link:** https://arxiv.org/html/2502.02309v1

**O que extrair:**
- A taxonomia em 3 estágios (pré-processamento / in-processing / pós-processamento) — você precisa **citar de cor** porque é a estrutura organizadora da literatura.
- A definição formal de Inequity Rate, FDR, GARBE, CEI — você usa estas no seu paper.
- A seção "Future Directions" — sua proposta se posiciona dentro de 2 dos eixos prioritários listados.

**Pergunta antecipada:** *"Como sua tese se diferencia do que já está na revisão de Kotwal?"*
**Resposta-âncora:** "A revisão consolida métricas e métodos; minha tese é uma **avaliação empírica controlada de duas intervenções específicas (loss-adaptive + dados sintéticos)** sobre o mesmo backbone, aplicado ao FairFace, e contribui com a evidência de que balanceamento sozinho deixa um gap residual sistemático mensurável."

#### 1.2 AdaFace — Kim et al., CVPR 2022 (~45 min)
**Link:** https://arxiv.org/abs/2204.00964 (paper) + https://github.com/mk-minchul/AdaFace (código)

**O que extrair:**
- A **intuição central**: a margem deveria ser **adaptativa à qualidade da imagem**. Imagens de baixa qualidade recebem margem menor (modelo não força a separação angular dura quando o dado é ruim).
- Como AdaFace usa a **norma do feature pré-classificação** como proxy de qualidade.
- Por que isso ajuda em fairness: classes minoritárias frequentemente têm imagens de qualidade variável; a margem adaptativa evita o colapso visto em ArcFace puro.

**Pergunta antecipada:** *"Por que AdaFace e não, por exemplo, MagFace ou KP-RPE?"*
**Resposta-âncora:** "AdaFace é a referência mais citada no eixo quality-adaptive (CVPR 2022, +900 citações), tem código público maduro, e aborda exatamente o modo de falha que minha smoke run expôs: o ArcFace 'real' colapsando classes minoritárias (Southeast Asian F1=0.107 no Exp 6). Vou adicionar MagFace como segunda condição para validar a robustez do achado."

#### 1.3 DCFace — Kim et al., CVPR 2023 (~40 min)
**Link:** https://arxiv.org/abs/2304.07060 + https://github.com/mk-minchul/dcface

**O que extrair:**
- A **arquitetura dual condition**: uma cabeça controla **identidade** (quem é a pessoa), outra controla **estilo** (pose, iluminação, idade aparente).
- **Como gerar uma classe específica** (ex.: Latino_Hispanic): condicionar com identidades reais dessa classe via embeddings ArcFace pré-treinados.
- O domain gap real → sintético (~2-3% nos benchmarks recentes) — isso é uma **limitação a reconhecer** no seu paper.

**Pergunta antecipada:** *"E se o DCFace gerar imagens enviesadas (que reforcem o estereótipo)?"*
**Resposta-âncora:** "Risco real, levantado em VariFace (2024). Mitigação: (a) condicionar o gerador em identidades reais da classe minoritária do FairFace (não amostragem livre do espaço latente), (b) avaliar a qualidade visualmente em uma amostra antes de incluir no treino, (c) o teste final é cego: as métricas (IR, gap) sobre o test set REAL determinam se a augmentação ajudou. Se não ajudar ou piorar, é resultado publicável."

#### 1.4 Buolamwini & Gebru — *Gender Shades*, FAT* 2018 (~30 min)
**Link:** https://proceedings.mlr.press/v81/buolamwini18a.html

**O que extrair:**
- O paper que **fundou** o campo de auditoria demográfica em FR. Seu trabalho descende diretamente disso.
- A metodologia: comparar performance entre subgrupos demográficos formados por raça × gênero (interseccional).
- A frase-chave: *"darker-skinned females are the most misclassified group"* — sua descoberta de que Latino_Hispanic é o pior é uma extensão moderna desse padrão.

**Pergunta antecipada:** *"Você cita Gender Shades?"*
**Resposta-âncora:** "É a referência fundadora do campo, citada na introdução e nas motivações regulatórias. Minha tese atualiza essa metodologia com métricas mais formais (IR/FDR/Gini) e investiga **intervenções de mitigação**, enquanto Gender Shades é majoritariamente diagnóstico."

---

### Bloco 2 (almoço/após, ~2h) — Releitura da sua própria proposta + dados

Releia a sua proposta **com olhar crítico**, simulando que você é o coordenador. Marque pontos vulneráveis. Em seguida, releia os 2 docs técnicos para ter os números na ponta da língua.

#### 2.1 Sua própria proposta (~30 min)
- [PROPOSTA_MESTRADO.md](PROPOSTA_MESTRADO.md)
- Marque com lápis o que você **não conseguiria explicar oralmente** sem hesitar. Esses são seus pontos de estudo extra.

#### 2.2 Smoke results (~30 min)
- [docs/smoke_results.md](smoke_results.md)
- **Memorize 5 números**:
  - Exp 1: acc=0.663, F1=0.662, **IR=1.73**, gap=0.34
  - Exp 6 (pior IR): IR=7.21, Southeast Asian F1=0.107
  - Exp 7 binário: 0.94 em 5 épocas
  - Exp 9 (dropout=0.5, melhor IR 7-class): IR=1.638
  - Exp 10 divergiu: val_acc=0.143 = 1/7

Estes números são suas **provas empíricas** — deveria recitá-los sem olhar.

#### 2.3 Exp 1 vs MBA (~30 min)
- [docs/exp01_vs_mba.md](exp01_vs_mba.md)
- Memorize:
  - MBA reportou acc=0.68, F1 macro=0.68 para Exp 1 (CE+SGD+OneCycleLR, 25 ép, dropout=0.2)
  - Nossa replicação atingiu acc=0.636, F1=0.634 (best.pt época 5, dropout=0.5)
  - Latino_Hispanic ficou em F1=0.433 (vs MBA 0.51)

#### 2.4 Estrutura do código (~30 min)
- Releia README.md
- Saiba **descrever em 30 segundos** o pipeline: download → preprocess (MTCNN) → dataset (PyTorch) → train (LResNet50E_IR + Loss + Scheduler) → evaluate (acc + fairness) → interpretability (t-SNE + Grad-CAM)

---

### Bloco 3 (tarde, ~2h) — As perguntas difíceis

Antecipe e ensaie respostas para as perguntas que provavelmente virão. Escreva 1 parágrafo de resposta para cada — **escrever obriga clareza**.

#### Q1. "Por que isso é mestrado e não engenharia?"

**Pontos da resposta:**
- Pergunta empírica testável (PQ1, PQ2, PQ3) com hipóteses falsificáveis (H1, H2, H3).
- Resultado de qualquer cenário (positivo ou nulo) é publicável.
- Achado já alcançado (H0 falsificada) é contribuição original validável.
- Conexão com literatura recente (Kotwal & Marcel 2025) e contexto regulatório (EU AI Act).

#### Q2. "O que de fato é novo aqui?"

**Pontos:**
- A maioria dos trabalhos de fairness em FR ou (a) propõe um novo método e mede em 1-2 setups, ou (b) faz survey/auditoria sem intervenção. **Sua tese faz auditoria + intervenção controlada no mesmo pipeline.**
- O bug §2.2 (ArcFaceLoss → cross-entropy) é um caso documentado de invalidação retroativa de resultados publicados localmente. Reportar isso é contribuição em reprodutibilidade.
- Métrica IR sobre o mesmo dataset balanceado, antes/depois de cada intervenção: **decomposição que poucos artigos fazem explicitamente**.

#### Q3. "Por que ResNet50 e não ViT?"

**Pontos:**
- Hardware local (RTX 4070 SUPER, 12 GB) não comporta ViT-L; ViT-B fica apertado.
- O paper foca em **intervenções comparáveis** sobre o mesmo backbone — trocar o backbone introduz variância confusora.
- TransFace++ é trabalho futuro (fase 2 pós-qualificação), já mapeado na proposta.

#### Q4. "E se nenhuma das duas intervenções fechar o gap?"

**Pontos:**
- **Null result é publicável** — vide tradições recentes em ML reproducibility (NeurIPS Datasets & Benchmarks, FAccT).
- Reframe: "limites das intervenções top-of-mind" + diagnóstico via t-SNE/Grad-CAM apontando que o problema é mais profundo (dataset bias, não classifier bias).
- Contribuição passa a ser negativa-mas-construtiva: aponta para direções que merecem mais atenção (BUPT-Balancedface, ViT, foundation models).

#### Q5. "Por que FairFace e não BUPT-Balancedface ou RFW?"

**Pontos:**
- FairFace é o ponto de partida do MBA, garantindo continuidade comparável.
- O paper está scopado para **uma intervenção controlada**, não cross-dataset evaluation.
- BUPT/RFW são **trabalho futuro** já planejado (fase 2). RFW exige aplicação institucional formal demorada.

#### Q6. "Quais métricas de fairness você usa, e por quê?"

**Pontos:**
- **Inequity Rate (IR)** — max(F1_class) / min(F1_class). Headline metric. Pereira & Marcel; Kotwal & Marcel (2025).
- **Max-min disparity** — gap absoluto, intuitivo.
- **Coefficient of Variation** — proxy do FDR para classificação.
- **Gini** — família GARBE.
- Reportar todos os 4 evita "metric hacking" (otimizar para uma e ignorar as outras).

#### Q7. "Você consegue chegar a um paper publicável em 6 meses?"

**Pontos:**
- Mês 1 já tem evidência empírica suficiente para o "Ato 1" do paper (diagnóstico).
- Meses 2-4 são experimentos focados (~6-9h GPU cada) — totalmente factível em RTX 4070 SUPER.
- Mês 5 dedicado à escrita + submissão (WACV Fair-CV deadline ~setembro/2026).
- Plano B: IJCB 2027 (deadline abril) se WACV não fechar.

---

### Bloco 4 (final do dia, ~1h) — Material visual e talking points

#### 4.1 Print das figuras-chave (~30 min)
Tenha em mãos (impresso ou em tablet):
- `outputs/figures/exp01/fairness.png` — gráfico de barras com IR=1.878 (essa é a imagem-chave do "ato 1")
- `outputs/figures/exp01/metricas.png` — mostra que fairness MELHORA com mais épocas (achado novo)
- `outputs/figures/exp01/matriz.png` — confusão entre classes asiáticas + Latino_Hispanic
- `outputs/tsne/<run>/tsne.png` — Latino_Hispanic disperso pelo plano

Estas 4 figuras contam toda a história do "ato 1".

#### 4.2 Talking points para a abertura (~30 min)
Escreva e ensaie um **pitch de 2 minutos** para começar a reunião:

> "Professor, fiz minha dissertação de MBA na USP em mitigação de viés em reconhecimento facial. A premissa era: balancear o dataset por undersampling produz reconhecimento equitativo. Quando refatorei o código e reexecutei os 11 experimentos do MBA com pipeline corrigido em maio/2026, **o resultado falsificou empiricamente essa premissa**: mesmo com 10 mil amostras por classe (todas as 7 idênticas), o modelo continua 73% mais propenso a acertar Black do que Latino_Hispanic. Esse achado é a base da minha proposta de mestrado: se balancear não basta, **o que basta?** Investigo duas frentes — funções de perda quality-adaptive e augmentação por dados sintéticos — em hardware local, com plano de submeter um paper no WACV Fair-CV Workshop até o quinto mês."

Decore essa abertura. Se você abrir bem, os primeiros 2 minutos dão o tom da reunião inteira.

---

## Checklist final (antes de dormir no domingo)

- [ ] Li e fichei Kotwal & Marcel (2025), AdaFace (Kim 2022), DCFace (Kim 2023), Gender Shades (Buolamwini 2018).
- [ ] Sei recitar IR=1.73, gap=0.34, F1 Latino_Hispanic=0.47, F1 Black=0.81 sem olhar nota.
- [ ] Tenho 4 figuras-chave em mãos para mostrar (fairness, métricas, confusão, t-SNE).
- [ ] Ensaiei o pitch de 2 minutos pelo menos 3 vezes em voz alta.
- [ ] Tenho a proposta impressa ou aberta em tela.
- [ ] Sei a diferença entre **PQ1, PQ2, PQ3** sem hesitar.
- [ ] Tenho resposta pronta para as 7 perguntas antecipadas (Q1–Q7).
- [ ] Sei o limite de hardware: ~34h GPU para 6 meses, RTX 4070 SUPER local.

---

## Material extra (caso tenha tempo)

Apenas se sobrar tempo no fim do dia:

- **MagFace** (Meng et al., CVPR 2021) — alternativa ao AdaFace, mesmo eixo. Skim de 15 min só para saber a intuição.
- **VariFace** (2024, arXiv:2412.06235) — sucessor do DCFace com foco em diversidade demográfica. Skim de 15 min.
- **EU AI Act Article 26** (texto regulatório, ~10 min) — saber que sistemas de alto risco precisam de auditoria documentada a partir de 02/08/2026.

Não obrigatório. Se chegar saturado, prefira **descansar** — chegar lúcido na reunião é mais importante do que ter lido um paper a mais.

---

## Posicionamento mental para a reunião

Três coisas que costumam impressionar coordenadores:

1. **Você sabe seus números.** Não precisa decorar a literatura inteira; precisa saber o que **você** mediu.
2. **Você antecipa as fraquezas.** Apresentar uma limitação antes do coordenador apontar é um sinal de maturidade.
3. **Você tem um plano B.** Para cada hipótese (H1, H2, H3) tenha pronta a resposta "se isso falhar, eu...".

Boa reunião!
