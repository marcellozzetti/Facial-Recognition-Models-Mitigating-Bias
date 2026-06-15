---
data: 2026-06-15
tipo: decisao-arquitetural
status: registrada
contexto: pos-reuniao-orientador-2026-06-15 + avaliacao-critica-mecanismos
---

# Decisão arquitetural — mecanismo de conditioning

> **Decisão registrada após avaliação crítica honesta dos candidatos
> a mecanismo de conditioning (FiLM, Concat, Conditional BN,
> Cross-attention, AdaIN, SPADE, HyperNet, LoRA, Adapter).**
>
> Avaliação considerou: sinal MST 10-dim, backbone ConvNeXt-T,
> restrição de orçamento (mestrado, 30 dias), interpretabilidade,
> lacuna na literatura de fairness.

## Escolha — FiLM como linha principal + Gated FiLM como ablation

### Linha principal (Configuração B do Cap 2 / 4)

- **Backbone**: ConvNeXt-T (CNN moderna sucessora dos ResNets)
- **Mecanismo**: **FiLM standard** (Perez et al., AAAI 2018)
  - `F' = γ · F + β` por canal
  - γ, β derivados de MLPs simples sobre o vetor MST 10-dim
- **Vantagens determinantes**:
  - Adequação ao sinal baixo-dim (10-dim)
  - Custo paramétrico ~1% sobre o backbone
  - Interpretabilidade direta de γ e β por canal
  - Compatibilidade com a estrutura LayerNorm do ConvNeXt-T
  - Lacuna na literatura: FiLM aplicado a fairness em race classification
    é direção pouco explorada — vira contribuição original

### Ablação metodológica obrigatória (Configuração C do Cap 2 / 4)

- **Gated FiLM** (variante não-linear):
  - `F' = sigmoid(γ_g) ⊙ tanh(γ_t) ⊙ F + β`
  - Testa se interações não-lineares trazem ganho sobre o afim simples
- **Justificativa para a banca**: mostra que o FiLM standard não foi
  escolhido por inércia — comparamos contra uma variante mais expressiva
  e medimos o trade-off

### Avaliação alternativa (Configuração D do Cap 2 / 4)

- **CLIP-conditioning** conforme orientação do orientador (reunião
  2026-06-15):
  - `F' = γ · F + β`, com γ e β derivados de MLPs sobre o embedding
    CLIP-text (512-dim)
- **Não é substituição** do FiLM-MST — é avaliação de **fonte alternativa
  de sinal semântico** mantendo o mesmo mecanismo de injeção

### Baseline (Configuração A)

- **ConvNeXt-T sem conditioning** — controle clássico para isolar o
  efeito do mecanismo

## Estudo comparativo final — 4 configurações no Capítulo 2

| Config | Mecanismo | Sinal | Papel |
|---|---|---|---|
| A | — | — | Baseline sem conditioning |
| **B** | **FiLM standard** | **MST 10-dim** | **Linha principal (nossa contribuição)** |
| C | Gated FiLM | MST 10-dim | Ablação — testa não-linearidade |
| D | FiLM standard | CLIP-text 512-dim | Avaliação alternativa (orientador) |

## Mecanismos avaliados e descartados — justificativa registrada

| Mecanismo | Por que descartado |
|---|---|
| Concat direto MST → backbone | Explode parâmetros, dilui sinal, sem modulação |
| Conditional BatchNorm | Caso particular do FiLM — generalizado pelo FiLM |
| Cross-attention | Overkill para sinal 10-dim (3× mais parâmetros que FiLM) |
| AdaIN | Voltado para style transfer; instance-level |
| SPADE | Requer mapa espacial denso (não vetor) |
| HyperNetworks | Instabilidade de treino; overkill |
| LoRA | Categoria diferente (fine-tuning de pesos, não conditioning de features) |
| Adapter layers | Sub-rede inserida, não é conditioning per se |

## Limitações honestas reconhecidas do FiLM

Para registro no Capítulo 4 (Discussão):

1. **Linear (afim por canal)** — não captura interações não-lineares
   entre tom de pele e race; **endereçado pela ablação Gated FiLM**.
2. **Não-espacial** — γ, β iguais em toda a imagem; coerente com a
   natureza global do skin tone, mas limita análises locais.
3. **Single-step** — modula uma vez, sem iteração de refinamento.

## Cenários em que outro mecanismo seria preferível (NÃO o nosso)

- Sinal é mapa espacial → SPADE
- Sinal é alto-dim sequencial (texto cru) → cross-attention
- Objetivo é fine-tuning de modelo grande → LoRA

Nenhum se aplica ao setup desta dissertação.

## Referência canônica

- **Perez, E.; Strub, F.; de Vries, H.; Dumoulin, V.; Courville, A.
  (2018).** *FiLM: Visual Reasoning with a General Conditioning Layer.*
  AAAI Conference on Artificial Intelligence. arXiv:1709.07871.
