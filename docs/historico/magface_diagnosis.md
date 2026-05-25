# Diagnóstico do colapso do MagFace (Fator 3) — investigação controlada

> Material de tese. Documenta uma investigação de depuração com método
> científico: hipóteses levantadas, evidência, falseamento, mecanismo
> final e a lição metodológica. Data: 2026-05-18.

## 1. Sintoma

No sanity de 2 épocas (protocolo idêntico ao dos demais Fatores: ResNet-50
pré-treinado, FairFace limpo, fp32, `num_workers=0`, AdamW lr=1e-3):

| head | acc (2ep) | F1 | `val_loss` ep1 / ep2 |
|---|---:|---:|---|
| AdaFace | 0.543 | 0.541 | 3.24 → 2.65 (desce) |
| **MagFace** | **0.151** | **0.065** | **1.94591 → 1.94591 (congelado)** |

`val_loss` ≡ **exatamente `ln(7) = 1.945907`** nas duas épocas e `val_acc`
≡ **exatamente `1/7`** — a saída de avaliação é literalmente uniforme
entre as 7 classes e invariante ao treino, enquanto `train_loss` desce.
AdaFace e ArcFace, no mesmo protocolo, treinam normalmente.

## 2. Hipóteses levantadas e falseadas

A investigação foi disciplinada: cada hipótese gerou uma predição
testável; três foram **falseadas** antes de chegar ao mecanismo real.

| # | Hipótese | Teste | Resultado |
|---|---|---|---|
| H1 | Falta do *guard* de monotonicidade (θ+m>π) | adicionar guard + re-sanity real | **Falseada** — colapso persistiu idêntico. (Guard é bug de correção real e foi mantido, mas dispara só para `cos<−0.9`, raríssimo no início → não é o mecanismo.) |
| H2 | Faixa `[l_a,u_a]=[10,110]` incompatível com a norma do backbone | probe da norma dos embeddings (treino e eval) | **Parcial/Falseada** — norma de treino ~6–14 (mediana 9,5), abaixo de `l_a`, mas margem resultante ≈0,45 (≈ ArcFace, que converge). Não explica o colapso sozinho. |
| H3 | Sem o regularizador `g(a)` (`lambda_g=0`) nada prende `‖f‖` | ligar `lambda_g=5` + diagnóstico instrumentado | **Falseada** — o regularizador *prende* a norma (‖f‖ ~10–11 em vez de cair a ~7), porém a representação ainda colapsa. Norma não era o mecanismo. |

Limitação reconhecida do instrumento: o diagnóstico de 160 passos
mostrava ArcFace *também* "colapsando" (`W_collapse→0.99`,
`logit_std→0.03`) — porém ArcFace treina bem em 25 épocas reais. Ou
seja, a janela curta captura uma instabilidade inicial comum a todas as
cabeças de margem; **não discrimina** a falha real. Conclusões só foram
tiradas do controle correto (§3) e do sanity real de 2 épocas.

## 3. Mecanismo real (com controle definitivo: AdaFace)

O controle ideal não é o ArcFace (instrumento curto, ambíguo) e sim o
**AdaFace**: mesmo backbone, mesmo protocolo, mesmo commit, **funciona**
(0,54 em 2 épocas reais). A única diferença estrutural no tratamento da
norma entre a cabeça que funciona e a que colapsa:

```python
# AdaFace (FUNCIONA) — adaface.py
norm = safe_norm.detach()        # "paper treats it as a statistic"

# MagFace (COLAPSAVA) — magface.py
a = raw_norm.clamp(l_a, u_a)     # NÃO detached -> m_a backpropaga por ||f||
```

**Mecanismo:** com `a` não-destacado, a margem `m(a)` é diferenciável em
relação a `‖f‖`. Isso abre um **atalho degenerado de gradiente**: o
otimizador reduz a loss *manipulando a norma para encolher a margem*, em
vez de aprender ângulos discriminativos. A representação colapsa, todos
os pesos de classe convergem para uma direção e a cabeça de avaliação
(cosseno escalado) vira saída uniforme → `val_loss = ln(7)` exato.
AdaFace **destaca** a norma justamente para impedir isto (comentário
explícito no código do paper); o MagFace não o fazia. `lambda_g` não
corrige porque o atalho é pela **margem**, independente do regularizador.

Por que só o MagFace, e não ArcFace/AdaFace:
- **ArcFace**: margem é constante (`math.cos(m)`), sem dependência de `‖f‖`.
- **AdaFace**: margem depende da norma, mas **destacada** (estatística).
- **MagFace (bug)**: margem dependia da norma **e era diferenciável**.

## 4. Correção (fiel ao paper e ao padrão que funciona)

Separação de responsabilidades exatamente como o paper do MagFace
prescreve:

1. **Margem `m(a)`** — magnitude tratada como *estatística*:
   `a = raw_norm.detach().clamp(l_a, u_a)`. Sem atalho de gradiente.
2. **Objetivo de magnitude diferenciável** — *somente* o regularizador
   canônico `λ_g · g(a)`, calculado na norma **bruta não-destacada**
   (`g(a)=1/u_a²·a + 1/a`, convexo, mínimo em `a=u_a` → o gradiente
   *puxa* `‖f‖` para cima). Cabeado no trainer (`_magface_reg`),
   somando `λ_g·last_g_reg` à loss de treino (não na avaliação).
3. `λ_g = 5` — calibrado ao regime de norma deste backbone (norma bruta
   ~6–14 → `g(a)` médio ~0,13; `λ_g=5` deixa o puxão ~10–30 % da CE; o
   35 do paper pressupõe a escala de magnitude do paper).

MagFace passa a ser avaliado na sua **forma canônica** — assim como
ArcFace e AdaFace nas formas canônicas deles.

### Testes de regressão (21/21 verdes)
- `test_magface_margin_is_magnitude_detached` — gradiente radial ≈0 com
  `λ_g=0` (trava o contrato do detach).
- `test_magface_regulariser_is_differentiable_and_lifts_norm` — `g`
  diferenciável e seu gradiente aumenta `‖f‖`.
- `test_magface_margin_never_increases_target_logit` — invariante do guard.

## 5. Lição metodológica (contribuição para a tese)

Este episódio **reforça a tese**, não a enfraquece:

> Famílias de loss de margem **não são livremente ablacionáveis**. O
> mecanismo do MagFace é inseparável do seu tratamento de magnitude
> (margem como estatística destacada + regularizador `g(a)`). Tentar
> reduzi-lo a "CE sobre logits de margem", como se faz com ArcFace, foi
> um **erro de desenho do Fator 3** que produziu um colapso — corrigido
> ao avaliar cada loss na sua configuração canônica.

Implicação para a decomposição causal: ao atribuir efeito a "família de
loss", cada loss deve entrar na **forma pretendida pelos autores**;
ablação ingênua confunde "efeito da loss" com "efeito de uma
configuração quebrada da loss". Esse é exatamente o tipo de armadilha
metodológica que o trabalho se propõe a expor (cf. `docs/sota_review.md`).

## 6. Scripts da investigação (reprodutíveis)
- `scripts/probe_embedding_norm.py` — distribuição da norma (treino/eval).
- `scripts/diag_magface_mechanism.py` — ablação sintética (inconclusiva, registrada por honestidade).
- `scripts/diag_magface_realtrain.py` — instrumentação do loop real (limitação da janela curta documentada aqui).
- Verdade-base: sanity real de 2 épocas (`outputs/factor3_sanity*`).
