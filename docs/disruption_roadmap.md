# Roadmap de disrupção — técnicas modernas candidatas (pós-qualificação)

> Documento de reflexão estratégica. Lista técnicas disruptivas modernas
> identificadas em 2026-05-23 que **NÃO entram na qualificação** mas
> podem ser executadas entre qualificação (ago/2026) e defesa final
> (data limite fev/2028) como expansões substanciais da tese.
> **Não rodar nada disso sem decisão consciente:** cada candidato adiciona
> 1-4 semanas de trabalho e tem risco de abrir nova frente experimental
> antes da redação dos capítulos atuais estar completa.
> Data: 2026-05-23. Status: **REFLEXÃO PENDENTE**.

## 1. Filtro de inclusão

Para entrar nesta lista, a técnica precisa:

1. **Ser moderna** (2020+, ainda ativa em 2024-2025).
2. **Ter literatura sólida** (pelo menos 1 paper em venue top — ICLR/NeurIPS/ICCV/CVPR/JMLR).
3. **Encaixar na nossa pipeline** sem reescrita massiva.
4. **Ter expectativa principiada** de mover F1 ou razão de disparidade.
5. **Avançar uma das contribuições declaradas** na `THESIS_STATEMENT.md §7`
   OU **fechar uma das limitações declaradas em §6** sem abrir nova.

## 2. Candidatos Tier 1 — alta disrupção, custo absorvível (1-2 semanas cada)

### 2.1 🚀 Group DRO — Linha B-bis (função de custo fairness-aware)

**Referência:** Sagawa et al., *Distributionally Robust Neural Networks for
Group Shifts: On the Importance of Regularization for Worst-Case
Generalization*, ICLR 2020 (arXiv 1911.08731). Vivo na literatura 2024
via extensões (Group DRO + Mixup, Group DRO + SAM).

**Mecanismo:**

$$\mathcal{L}_{\text{DRO}} = \max_{g \in \text{grupos}} \mathbb{E}_{x \sim P_g}[\ell(f_\theta(x), y)]$$

Minimiza a pior perda entre grupos em vez da média. Tem garantia teórica
de melhorar pior-grupo (Sagawa §3).

**Encaixe na nossa pipeline:**
- Substitui `CrossEntropyLoss` por `GroupDROLoss` em `src/face_bias/training/trainer.py`
- Implementação canônica em ~80 linhas (manter pesos exponenciais por grupo + atualizar via mirror descent)
- Não muda arquitetura nem dados

**Custo:**
- Implementação: ~4-6h
- Validação: ablação sob protocolo 🅔, 3 seeds × 2 arms = 6 runs × ~6h = ~36h GPU
- Documentação: ~3h
- **Total: ~1-2 semanas**

**Expectativa fundamentada na literatura:**
- IR: melhora 10-30% (Sagawa Tab.2 em CivilComments e CelebA)
- F1 macro: estável ou +1-3pp (pior-grupo sobe sem perder melhor-grupo)
- **Provavelmente melhora simultaneamente acurácia + equidade**, ao contrário do TrivialAugment

**Por que é tese-relevante:**
- **Linha B-bis (contribuição metodológica)** — testa empiricamente se loss fairness-aware
  supera a alavanca arquitetural (ConvNeXt-T) sob protocolo idêntico
- Adiciona uma 4ª contribuição: *"além da atribuição entre fatores algorítmicos
  intrínsecos, demonstramos que loss fairness-aware é alavanca complementar
  com magnitude X sob protocolo idêntico"*

**Risco:**
- Se Group DRO MOVER significativamente o IR (~−0.2 ou mais), **muda a
  narrativa central** da tese: "ConvNeXt é alavanca arquitetural; Group
  DRO é alavanca de loss; ambos contribuem cumulativamente". Forte.
- Se NÃO mover, ainda é null bem-medido, defensável.

### 2.2 🚀 DINOv2 backbone — Fator 6 (paradigma de pretraining)

**Referência:** Oquab et al. (Meta AI), *DINOv2: Learning Robust Visual
Features without Supervision*, arXiv 2304.07193 (2023). Extensão de
DINO (Caron 2021).

**Mecanismo:**

Self-supervised pretraining (sem rótulos humanos) em 142M imagens
curadas. Features universais que **não memorizam rótulos demográficos**
da fase de pretraining — argumentação central de Tian et al. 2024
(*"SSL features show less demographic shortcut learning than supervised
ImageNet features"*).

**Encaixe:**
- Adicionar `"dinov2_vitb14"` ao `build_backbone()` factory
- Usar via `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")`
- Embed dim = 768 (igual ViT-B/16); image_size = **518×518** (DINOv2 default) ou crops menores
- Pode usar via `transformers.AutoModel.from_pretrained("facebook/dinov2-base")`

**Custo:**
- Integração: ~3h
- Validação: 3 seeds × ~10h cada = ~30h GPU (DINOv2 é mais pesado por época)
- Documentação: ~3h
- **Total: ~1-2 semanas**

**Expectativa fundamentada:**
- F1: +1-3pp (DINOv2 é mais forte que ImageNet supervised na maioria dos benchmarks)
- IR: **provavelmente significativamente melhor** (SSL evita shortcuts demográficos)
- Caveat: ConvNeXt-T já é nossa alavanca; DINOv2 pode amplificar ou mudar a história

**Por que é tese-relevante:**
- **Adiciona Fator 6 (paradigma de pretraining: supervised ImageNet vs self-supervised DINOv2)** à matriz de atribuição causal
- Se DINOv2 > ConvNeXt-T em F1+IR → **nova alavanca descoberta**, contribuição forte
- Se DINOv2 ≈ ConvNeXt-T → null bem-medido, ainda informativo
- Se DINOv2 < ConvNeXt-T → mostra que ImageNet supervised não é o gargalo

**Risco:**
- Mudar arquitetura pode interagir com nosso recipe (lr, batch, etc.) — pode precisar nova rodada de sanity
- Sob protocolo 🅔 (já fechado), comparação válida

## 3. Candidatos Tier 2 — alta disrupção, ALTO custo (mês+ cada)

### 3.1 🚀🚀 Distillation from VLM (PaliGemma → ConvNeXt)

**Referência:** Hinton et al. (origem), 2015. Modernizado para
multimodal em 2023+: Wang et al. *Distilling CLIP for Specialist Models*
(ICLR 2024).

**Mecanismo:**

PaliGemma (~3B params, SOTA) gera predições sobre imagens FairFace;
nosso ConvNeXt-T treina para minimizar:

$$\mathcal{L} = (1-\alpha)\cdot \text{CE}(\hat{y}, y_{\text{true}}) + \alpha \cdot T^2 \cdot \text{KL}(\sigma(\hat{z}/T), \sigma(z_{\text{teacher}}/T))$$

**Por que é disruptivo:** **literalmente fecha o gap −5pp absoluto** para
SOTA-VLM ao transferir conhecimento sem precisar do modelo gigante em inferência.

**Custo:** ALTO
- Setup PaliGemma local ou via Vertex AI: ~2 dias
- Gerar predições sobre 97k FairFace: ~1 dia GPU
- Modificar trainer para KL-divergence + soft logits: ~1 dia
- Validação: 3 seeds × ~10h = ~30h GPU
- **Total: ~3-4 semanas**

**Risco:** ALTO — desvia foco da contribuição central (atribuição) para "competir
no número absoluto". Pode ser percebido como "saída da rota" pela banca.

### 3.2 🚀🚀 Stable Diffusion synthetic minority augmentation

**Referência:** Friedrich et al., *Fair Diffusion: Instructing Text-to-Image
Generation Models on Fairness*, arXiv 2302.10893 (2023).

**Mecanismo:** gerar amostras sintéticas das classes minoritárias via SDXL
condicionado em "face of [race] person, neutral expression, frontal pose".

**Por que disruptivo:** resolve o desequilíbrio sem perder dados (vs undersample)
ou duplicar (vs oversample) — gera amostras *novas* plausíveis.

**Custo:** MUITO ALTO
- Setup SDXL + fine-tuning para faces: ~1 semana
- Curadoria de prompts + filtragem: ~1 semana
- Validação rigorosa (não pode introduzir bias do próprio SDXL): ~1 semana
- **Total: ~1 mês**

**Risco:** ALTO — SDXL tem viés conhecido (gera White predominantemente);
"Fair Diffusion" mitiga parcialmente mas não é trivial. Pode introduzir
novo viés enquanto resolve o existente. **NÃO recomendado sem auditoria
empírica extensa da própria geração.**

## 4. Candidatos Tier 3 — baixa disrupção, baixíssimo custo (horas-dias)

Estes JÁ ESTÃO PLANEJADOS no combo defesa-fechamento (ver
`THESIS_STATEMENT.md §4.4` e `anchors_results.md §6.5`):

- **Análise interseccional** (race × gender × age) — ~30min, zero GPU
- **Ensemble de 3 seeds** (média de logits) — ~30min, zero GPU
- **TTA (Test-Time Augmentation)** — ~2h, zero treino
- **Calibração + threshold optimization per-class** — ~30min, zero GPU

Estes são **execução padrão**, não disrupção.

## 5. Recomendação estratégica para a tese

### Para a qualificação (ago/2026)

**NÃO rodar nenhum candidato deste roadmap.** A qualificação tem:
- 5 fatores + 3 anchors + ablação 🅑 + anchor 🅔 + auditoria empírica (testes A/B/C1) + ablação D (oversample)
- 3 contribuições defensáveis: atribuição causal, Pareto-aware criterion, auditoria de confundidores
- Combo defesa-fechamento Tier 3 (~3h adicional) fecha as últimas vulnerabilidades

Esse pacote satisfaz padrões de mestrado. Adicionar Group DRO ou DINOv2
ANTES da qualificação **abriria nova frente experimental sem fechar
a vulnerabilidade que já está coberta** — violaria a régua de decisão
da `THESIS_STATEMENT.md §11`.

### Entre qualificação e defesa final (ago/2026 — fev/2028)

**Janela de ~18 meses.** Aqui Tier 1 é elegível:

| Janela | Candidato | Justificativa estratégica |
|---|---|---|
| Out-Dez 2026 | **Group DRO** (Linha B-bis) | Linha B (Pareto-aware) é a contribuição metodológica atual; estender para Group DRO transforma em "auditoria + correção de critério + alavanca de loss fairness-aware" — contribuição forte |
| Jan-Mar 2027 | **DINOv2 backbone** (Fator 6) | Estende matriz de atribuição com paradigma de pretraining — modernização natural |
| Abr-Jun 2027 | Cross-dataset eval (RFW/DemogPairs) | Generalização — padrão de defesa |
| Jul-Set 2027 | Tier 3 polimento + redação final | — |
| Out-Dez 2027 | Tier 2 (distillation) **OPCIONAL** | Só se quiser tentar fechar gap absoluto vs VLM SOTA |
| Jan-Fev 2028 | Defesa | — |

### Critério para promover Tier 1 → execução

**Antes de iniciar Group DRO ou DINOv2 pós-qualificação, verificar:**

1. Qualificação aprovada com feedback positivo sobre Linha B (validação da abordagem metodológica).
2. Tempo disponível ≥ 6 semanas dedicadas (não dividido com outros compromissos).
3. **A contribuição esperada avança §7 ou fecha §6** (régua da `THESIS_STATEMENT.md §11`).
4. Não há sinal da banca de qualificação pedindo CROSS-DATASET ou outro item de prioridade superior.

## 6. Status atual

**Tudo neste documento está PENDENTE DE REFLEXÃO.** Não iniciar
implementação sem decisão consciente pós-qualificação. Atualizar
quando houver feedback da banca de qualificação ou novos achados que
mudem a priorização.

## 7. Procedência (papers candidatos)

| Técnica | Referência principal | Venue/ano |
|---|---|---|
| Group DRO | Sagawa, Koh, Hashimoto, Liang | ICLR 2020 (arXiv 1911.08731) |
| DINOv2 | Oquab et al. (Meta AI) | arXiv 2304.07193 (2023) |
| Fair Mixup | Chuang & Mroueh | ICLR 2021 (arXiv 2103.06503) |
| SAM | Foret, Kleiner, Mobahi, Neyshabur | ICLR 2021 (arXiv 2010.01412) |
| Fair Diffusion | Friedrich et al. | arXiv 2302.10893 (2023) |
| CLIP feature audit fairness | Sun et al. | 2024 (vários papers) |
| SSL features less biased | Tian et al. 2024, Mai et al. 2024 | venues variados |
| Distillation modernized | Hinton 2015 (origem); Wang 2024 (modernização VLM) | ICLR 2024 |
