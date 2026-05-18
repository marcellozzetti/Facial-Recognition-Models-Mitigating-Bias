# Resumo de progresso — semana de 12 a 16/05/2026

**Para:** Prof. Marcos Quiles
**De:** Marcello Ozzetti
**Assunto:** Atualização semanal (sessão de 18/05 remarcada) — objetivo, testes de MLP, achados e próximos passos

---

Prof. Marcos, segue um resumo objetivo do que avancei nesta semana, já
que nossa sessão foi remarcada. Organizei em quatro blocos: **objetivo,
testes de MLP, achados defensáveis e próximos passos**, e deixo **uma
pergunta** que destrava trabalho enquanto não nos falamos.

## 1. Objetivo do trabalho — travado e validado

Após revisão sistemática + auditoria semântica de **555 artigos** (as 76
referências do FairFace + 479 que o citam, via OpenAlex), conclui que
"mitigação de viés" e mesmo "busca multi-objetivo de arquitetura para
fairness" são espaço saturado (NeurIPS 2023, FairGRAPE/ECCV 2022,
FineFACE 2024). Reposicionei o objetivo de *mitigação* para
**atribuição causal**:

> Quantificar a contribuição marginal de cada fator do pipeline
> (dataset, topologia do classificador, função de perda, paradigma de
> aprendizado, backbone) para a disparidade demográfica — responder
> **onde intervir**, não *se* dá para melhorar.

O delta defensável, sem método equivalente nos 555 artigos: um
**critério Pareto-aware** de seleção em HPO multi-objetivo + a
**decomposição experimental controlada** — lacuna explicitamente
apontada como problema aberto no survey de fairness de 2025.

## 2. Testes de MLP no classificador — critério, execuções, achados

Seguindo a sua orientação de sair do classificador linear, implementei
um **cabeçote MLP configurável** entre o backbone ResNet-50 e a saída,
com profundidade, largura por camada, ativação, normalização e dropout
parametrizáveis.

**Critério de busca.** Em vez de testar topologias à mão, usei **Optuna
em modo multi-objetivo** (maximizar F1 macro **e** minimizar Inequity
Rate simultaneamente), recuperando a **frente de Pareto** entre
utilidade e equidade. Ponto metodológico relevante: o critério ingênuo
de selecionar a época de "melhor F1" dentro de cada trial descarta
épocas Pareto-ótimas; adotei um **critério Pareto-aware** (época
não-dominada, desempate pela menor disparidade). É o núcleo da
contribuição metodológica.

**Execuções.** (i) Estudo Optuna sobre o dataset limpo — 20 trials × 8
épocas, frente de Pareto com 2 topologias vencedoras; (ii) **refit de
confirmação** dos vencedores do Pareto em **25 épocas × 3 seeds** (a
busca curta não basta como evidência — só o refit de orçamento completo
confirma).

**Achados.** A topologia vencedora é simples e robusta:
`[256] GELU dropout=0,52`. As topologias **profundas com dropout baixo**
pareciam ótimas na busca curta mas **não se sustentaram** no refit
completo — exatamente o que justifica a necessidade do critério
Pareto-aware + etapa de confirmação obrigatória (busca curta de fairness
é não-confiável; isso vira contribuição metodológica, não só um detalhe).

## 3. Decomposição de fatores — resultados defensáveis

Protocolo adotado para **todos** os experimentos: **3 seeds (42,1,2),
base de comparação casada, ambiente único, média ± desvio**; ganho
dentro de um desvio-padrão é declarado não-significativo. Sob esse
protocolo, 2 dos 5 fatores já estão medidos de forma defensável:

| Fator | → Acurácia (F1) | → **Equidade (Inequity Rate)** |
|---|---|---|
| **Dataset** (limpeza de imagens multi-face, MTCNN) | **+1,35 pp** (CE, significativo) | nula (não-significativa, ambos recipes) |
| **Topologia** (linear → MLP `[256] GELU dropout=0,52`) | +0,8 pp (n.s.) | **−0,11 (estatisticamente significativo)** |

**Conclusão central:** a alavanca de equidade defensável é a
**topologia do classificador** — a limpeza do dataset contribui para
**acurácia**, não para fairness. Esse contraste (cada fator paga em um
eixo diferente) é precisamente o tipo de resultado que a metodologia de
atribuição existe para produzir.

Achado lateral robusto: a taxa de imagens com múltiplas faces no
FairFace é **correlacionada com raça** (spread de 9,3 pp entre grupos)
— viés de cena, além do desbalanceamento de rótulo.

## 4. Próximos passos

- **Fatores 3-5** (função de perda quality-adaptive, aprendizado
  contrastivo, backbone) — medidos um a um no mesmo protocolo
  controlado de 3 seeds (cronograma: junho/julho).
- **Verificação final de novidade:** leitura de 5 artigos sinalizados
  pela auditoria semântica como os mais próximos do delta (em
  andamento).
- Consolidação do mapa de atribuição (5 fatores) → núcleo da
  qualificação (alvo 24/08).

## 5. Pergunta que destrava trabalho

Na última conversa o senhor mencionou **"função de pesos"**. Quero
confirmar a interpretação: o senhor se referia a **loss ponderada por
classe** (weighted loss) como alternativa/complemento ao
*undersampling*? Se for, é uma **alavanca de balanceamento que o MBA
nunca testou** e encaixa diretamente como um fator novo na decomposição
(undersampling vs weighted-loss vs nenhum). Tenho a configuração pronta
para incluí-la assim que o senhor confirmar.

Fico à disposição. Código, resultados e documentação estão versionados
e reproduzíveis no repositório.

Atenciosamente,
Marcello
