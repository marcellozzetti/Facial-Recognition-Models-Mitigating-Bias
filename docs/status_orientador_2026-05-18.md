# Resumo de progresso — semana de 12 a 16/05/2026

**Para:** Prof. Marcos Quiles
**De:** Marcello Ozzetti
**Assunto:** Atualização semanal (sessão de 18/05 remarcada) — ações, achados e próximos passos

---

Prof. Marcos, segue um resumo objetivo do que avancei nesta semana, já
que nossa sessão de hoje foi remarcada. Sintetizei em três blocos:
**(1) decisão de objetivo, (2) achados, (3) próximos passos**, e deixo
**uma pergunta** que destrava trabalho enquanto não nos falamos.

## 1. Objetivo do trabalho — travado e validado

Após revisão sistemática + auditoria semântica de **555 artigos** (as 76
referências do FairFace + 479 que o citam, via OpenAlex), conclui que
"mitigação de viés" e mesmo "busca multi-objetivo de arquitetura para
fairness" **já são espaço saturado** (NeurIPS 2023, FairGRAPE/ECCV 2022,
FineFACE 2024). Reposicionei o objetivo de *mitigação* para
**atribuição causal**:

> Quantificar a contribuição marginal de cada fator do pipeline
> (dataset, topologia do classificador, função de perda, paradigma de
> aprendizado, backbone) para a disparidade demográfica — responder
> **onde intervir**, não *se* dá para melhorar.

O delta defensável (sem método equivalente nos 555 artigos): um
**critério Pareto-aware** de seleção em HPO multi-objetivo + a
**decomposição experimental controlada** — lacuna explicitamente
apontada como problema aberto no survey de fairness de 2025.

## 2. Achados (2 dos 5 fatores já medidos com rigor)

Adotei como padrão fixo: **3 seeds (42,1,2), base de comparação casada,
ambiente único, média ± desvio**. Isso revelou que conclusões
preliminares baseadas em 1-seed estavam **erradas por confound**
(versão de framework misturada com o efeito do dataset). Corrigi com um
experimento controlado de 12 execuções. Resultados defensáveis:

| Fator | → Acurácia (F1) | → **Equidade (Inequity Rate)** |
|---|---|---|
| **Dataset** (limpeza de imagens multi-face, MTCNN) | **+1,35 pp** (CE, significativo) | **nula** (não-significativa, ambos recipes) |
| **Topologia** (classificador linear → MLP otimizado) | +0,8 pp (n.s.) | **−0,11 (estatisticamente significativo)** |

**Conclusão central:** a alavanca de equidade defensável é a
**topologia do classificador**, **não** a limpeza do dataset (a limpeza
melhora acurácia, não fairness). Importante: a metodologia de
decomposição **detectou e descartou um efeito falso** (uma análise
anterior, sem rigor, sugeria −54% de IR via limpeza — era artefato) e
**revelou um efeito real** que a análise frágil havia descartado.
Isso será reportado como subseção "ameaças à validade e correções" — é
o próprio método de atribuição se provando.

Achado lateral robusto: a taxa de imagens com múltiplas faces no
FairFace é **correlacionada com raça** (spread de 9,3 pp entre grupos)
— viés de cena, além do desbalanceamento de rótulo.

## 3. Próximos passos

- **Fatores 3-5** (função de perda quality-adaptive, aprendizado
  contrastivo, backbone) — medidos um a um no mesmo protocolo controlado
  de 3 seeds (cronograma: junho/julho).
- **Verificação final de novidade:** leitura de 5 artigos sinalizados
  pela auditoria semântica como os mais próximos do delta (em
  andamento).
- Consolidação do mapa de atribuição (5 fatores) → núcleo da
  qualificação (alvo 24/08).

## 4. Pergunta que destrava trabalho

Na última conversa o senhor mencionou **"função de pesos"**. Quero
confirmar a interpretação: o senhor se referia a **loss ponderada por
classe** (weighted loss) como alternativa/complemento ao
*undersampling*? Se for, ela é uma **alavanca de balanceamento que o
MBA nunca testou** e encaixa diretamente como um fator novo na
decomposição (undersampling vs weighted-loss vs nenhum). Tenho a
configuração pronta para incluí-la assim que o senhor confirmar.

Fico à disposição. Os artefatos (código, resultados, documentação) estão
versionados e reproduzíveis no repositório.

Atenciosamente,
Marcello
