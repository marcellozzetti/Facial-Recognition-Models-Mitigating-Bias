# Teste de mesa bit a bit — fórmulas (métricas + arquiteturas) vs literatura

> Material de tese. Verificação linha a linha das fórmulas contra as
> equações dos papers de origem. Pedido: "não posso ter mais erro".
> Data: 2026-05-18.

Severidade: 🔴 corrige interpretação/honestidade · 🟠 fidelidade de
hiperparâmetro · 🟢 correto.

## 1. Métricas (`evaluation/metrics.py`)

| Métrica | Fórmula no código | Verificação | Veredito |
|---|---|---|---|
| accuracy / f1_macro / f1_weighted | sklearn, `average="macro"`, `zero_division=0` | padrão | 🟢 correto |
| log_loss | sklearn, `labels=range(C)` | padrão | 🟢 correto |
| max_min_disparity | `max − min` | gap absoluto padrão | 🟢 correto |
| gini | `(n+1 − 2·Σcum/cum[-1])/n` sobre ordenado | **provado algebricamente** = `2Σi·xᵢ/(nΣx) − (n+1)/n`, a forma fechada canônica de Gini | 🟢 correto |
| coefficient_of_variation | `std(ddof=0)/mean` | CV correto **mas** docstring diz "FDR-style" — FDR (Fairness Discrepancy Rate, Pereira&Marcel) **não é** CV | 🟠 nome enganoso |
| **inequity_rate** | `max(score)/min(score)` por classe | O IR canônico (Pereira&Marcel, verificação) é **produto de razões FMR e FNMR**. Nosso uso é uma **adaptação** (razão max/min de F1 por classe) — defensável, mas **não é** o IR do paper | 🔴 terminologia |

**Conclusão métricas:** numericamente **corretas** (Gini provado à mão).
O problema é **fidelidade de nomenclatura**: `inequity_rate` e o rótulo
"FDR-style" invocam construtos da literatura de verificação que as
fórmulas não implementam literalmente. Para a banca: ou renomear
(ex.: "max-min ratio" / "disparity ratio") ou definir explicitamente
como adaptação, com a fórmula declarada no texto. **Não invalida os
números já obtidos** (a comparação é interna e consistente), mas
invalida a *alegação* de que medimos "o Inequity Rate da literatura".

## 2. Cabeças de margem

### 2.1 ArcFace (`arc_margin.py`) — Deng et al., CVPR 2019
- `phi = cos·cos_m − sin·sin_m` = `cos(θ+m)` ✓ exato.
- guard `where(cos>th, phi, cos−mm)`, `th=cos(π−m)`, `mm=sin(π−m)·m`
  = implementação oficial insightface (`easy_margin=False`) ✓.
- 🟠 **`s` default = 30; o paper ArcFace usa `s=64`.** `s=30` é
  defensável p/ 7 classes/regime de norma menor, mas **é desvio do
  canônico** — declarar no texto. Fórmula: 🟢 correta.

#### Racional científico de `s=30` (decisão travada: manter 30, documentado)

`s` é o raio da hiperesfera / temperatura inversa do softmax sobre o
cosseno normalizado. O `s=64` do paper ArcFace **não** é universal — é
co-ajustado a verificação facial com **dezenas de milhares de
identidades** (MS1M ~85k classes), embeddings 512-d L2-norm de backbones
IR treinados do zero. Três razões pelas quais `s=30` é a escolha
**cientificamente correta** aqui, não apenas conveniente:

1. **`s` mínimo escala com o nº de classes `C`.** O limiar teórico para
   o softmax de margem saturar na classe correta (Zhang et al.,
   *Heated-up Softmax*; apêndice do CosFace) cresce com `C`. Para
   `C=85k` exige-se `s` grande (~64); para **`C=7`** o `s` necessário p/
   atingir confiança ~0,9 é da ordem de **~10**. `s=30` já é folgado p/
   7 classes; **`s=64` com 7 classes super-satura** o softmax (gradiente
   ~0 fora da fronteira) e **prejudica a otimização**. Logo `s=30` é
   *mais apropriado* ao baixo `C`, não um compromisso.
2. **Transferir `s=64` seria transferência injustificada de regime**:
   ele foi co-ajustado a 512-d/IR/scratch/85k-id; nada disso vale para
   um classificador de atributo 7-classes sobre ResNet-50 ImageNet. Não
   há base para herdar o `s` canônico fora do regime canônico.
3. **Casamento experimental (regra permanente do projeto).** MBA,
   `dataset_factor`, `factor3` e HPO todos usam `s=30`. Mudar `s` no
   meio introduziria um confound idêntico aos já capturados (R1/R2,
   critério de checkpoint). A metodologia de atribuição causal exige
   `s` **fixo e igual** em todos os fatores; `s` não é o fator sob
   estudo no Fator 3 (a *família de loss* é).

➡️ **Decisão: `s=30` mantido, com este racional declarado na
dissertação.** Severidade 🟠 → resolvida/documentada.

### 2.2 AdaFace (`adaface.py`) — Kim et al., CVPR 2022
- `ẑ = clip((‖f‖−μ)/(σ/h), −1, 1)`, `g_angle=−m·ẑ`,
  `g_add=m·ẑ+m`, `target=cos(θ+g_angle)−g_add` ✓ fiel ao paper.
- `‖f‖.detach()` (estatística), EMA momentum 0.99, h=0.333, m=0.4 ✓.
- 🟠 **Init dos buffers EMA `batch_mean=20.0, batch_std=100.0`** está
  **longe do regime real** (probe: ‖f‖≈9±1,6). Com momentum 0,99 a EMA
  só converge após ~centenas de batches → por muitas épocas `ẑ≈0` →
  margem ≈ constante (`−m` no cosseno), comportamento adaptativo
  atrasado. Não é bug de fórmula, mas **degrada o que o AdaFace promete**
  no início do treino. Recomendo init próximo do regime medido
  (ex.: mean≈9, std≈2) ou warmup mais rápido. Fórmula: 🟢 correta.

### 2.3 MagFace (`magface.py`) — Meng et al., CVPR 2021
- `m(a)` linear crescente em `a` ✓ (maior magnitude → maior margem).
- `g(a)=(1/u_a²)·a + 1/a`, decrescente em [l_a,u_a] → `−∇` puxa ‖f‖
  para cima ✓; aplicado na norma **bruta, diferenciável**; margem usa
  `a` **clampado e detached** ✓ (separação fiel ao paper; ver
  [magface_diagnosis.md](magface_diagnosis.md)). Fórmula: 🟢 correta
  (pós-fix).

## 3. Backbone — RESOLVIDO: 🔴 → 🟠 (terminologia, não confound)

> **Atualização (2026-05-18, pós-verificação tripla).** Recuperado o
> código real do MBA: `old/models.py` (histórico git `cbf3bfd^`) e o
> link do usuário (commit `edb098c`, confirmado byte a byte). O
> `LResNet50E_IR` do MBA **já era** `torchvision.resnet50(ImageNet)` +
> `fc=Identity` + `Dropout(0.2)` + `nn.Linear` — **idêntico** ao
> backbone do `src/` atual. **Não há mismatch nem confound
> MBA↔mestrado**; a comparação está casada no eixo backbone. O nome
> "LResNet50E-IR" é um misnomer **consistente**, herdado do MBA.
> Bônus: no MBA o `ArcMarginProduct` estava definido mas **nunca
> conectado** → "ArcFace do MBA" era Linear+CE (diferença de *loss*,
> estudada no Fator 3; não de backbone).
>
> **Decisão do usuário: Recomendação A aprovada** — manter ResNet-50
> ImageNet (preserva casamento + toda a base experimental), **renomear**
> a classe + documentar o regime, e tratar IR-real/ViT como o **eixo de
> backbone do Fator 5** (PLANO). Severidade rebaixada 🔴→🟠.

### (registro original do achado, mantido para rastreabilidade)

A classe chama-se **`LResNet50E_IR`**. Na literatura de reconhecimento
facial (ArcFace/insightface) esse nome designa uma arquitetura
**específica**: ResNet-50 com blocos **IR (Improved Residual)**, stem
modificado, entrada **112×112**, saída **BN-Dropout-FC-BN → embedding
512-d**, treinada do zero em milhões de faces.

**O código implementa:** `torchvision.models.resnet50(weights=ImageNet)`
com `fc=Identity` → **ResNet-50 ImageNet padrão**, entrada **224×224**,
embedding **2048-d**, transfer learning.

➡️ **Não é LResNet50E-IR.** É ResNet-50 ImageNet. O nome **alega uma
arquitetura da literatura que não está implementada.** Isto é
exatamente "não fazer o que a literatura propõe":

- **Não invalida os experimentos numericamente** (ResNet-50 ImageNet é
  backbone válido e o uso é interno-consistente), **mas invalida a
  descrição** se a dissertação disser "LResNet50E-IR".
- **Implicação científica real:** ArcFace/AdaFace/MagFace foram
  desenhadas para embeddings 512-d L2-normalizados de backbones IR
  @112px treinados do zero em escala. Aplicá-las sobre features 2048-d
  ImageNet @224px num softmax de 7 classes é um **regime muito fora**
  do dos papers — o que **explica fisicamente** por que as losses de
  margem não superam CE aqui (achado empírico ↔ regime de operação).

**Ação mínima obrigatória p/ a tese:** parar de chamar de
`LResNet50E_IR`; documentar como "ResNet-50 (ImageNet) + cabeça X,
transfer learning, 224px, embedding 2048-d". Opcional/forte: implementar
de fato o backbone IR @112px se quiser comparar no regime dos papers
(eixo backbone do PLANO).

## 4. Veredito consolidado

| Item | Numérico | Fidelidade à literatura |
|---|---|---|
| Métricas | 🟢 corretas (Gini provado) | 🔴 `inequity_rate`/"FDR" mal-nomeados |
| ArcFace | 🟢 | 🟠 s=30 vs 64 canônico |
| AdaFace | 🟢 | 🟠 init EMA fora do regime |
| MagFace | 🟢 (pós-fix) | 🟢 |
| **Backbone** | 🟢 (válido) | 🔴 **nome ≠ implementação** |

**Não há novo erro de cálculo.** Os achados são de **fidelidade à
literatura/nomenclatura** — e dois (🔴) afetam diretamente a
*descrição defensável* na qualificação: (i) o backbone não é o que o
nome diz; (ii) "Inequity Rate" é uma adaptação, não o IR de
verificação. Ambos têm correção barata (renomear + documentar) e, na
verdade, **reforçam a Linha A**: parte do "viés" atribuído a fatores
pode ser efeito de *regime fora do paper*, exatamente o tipo de
confusão causal que a tese expõe.
