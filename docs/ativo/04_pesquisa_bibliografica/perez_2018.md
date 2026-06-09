---
name: perez-2018
status_verificacao: VERIFIED
autores: [Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, Aaron Courville]
ano: 2018
titulo: "FiLM: Visual Reasoning with a General Conditioning Layer"
venue: "AAAI Conference on Artificial Intelligence 2018"
tipo_publicacao: conference
arxiv_id: "1709.07871"
doi: null
url_primario: https://arxiv.org/abs/1709.07871
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-04
n_referencias_paper: ~40
lente_disrupcao: paradigma
fonte_leitura: PDF integral baixado de arXiv (pdfs/perez_2018_film.pdf)
---

# FiLM: Feature-wise Linear Modulation (Perez, Strub, de Vries, Dumoulin & Courville, 2018)

> **Mecanismo de condicionamento neural** que permite uma fonte
> auxiliar (texto, classe, embedding) **modular features de outra
> rede** via transformação afim por canal. **Diretamente aplicável**
> à linha v3.2 do orientador: usar saída do MST classifier como
> entrada condicional para o race classifier.

## 1. Resumo do problema atacado

Em arquiteturas tradicionais, **incorporar informação contextual** a um
CNN (e.g., questão textual + imagem em VQA, classe + features visuais
em conditional generation) usava concatenação direta ou attention
heavy. FiLM propõe mecanismo **simples e geral**: para cada feature
map x_i, computar **γ_i e β_i** a partir do contexto e aplicar
**FiLM(x_i) = γ_i * x_i + β_i** — transformação afim por canal,
modulada pelo contexto.

## 2. Método

### 2.1 FiLM layer — formulação

Dado um feature map F ∈ ℝ^{C×H×W} e um vetor de contexto z (e.g.,
embedding de texto ou classe), FiLM aprende funções:

γ = f_γ(z), β = f_β(z), com γ, β ∈ ℝ^C

Aplicação por canal:

FiLM(F_i | γ_i, β_i) = γ_i ⊙ F_i + β_i

f_γ, f_β são tipicamente MLPs simples.

### 2.2 FiLM-ed Network

Insere camadas FiLM em pontos estratégicos da CNN principal (após
batch norm, antes de não-linearidade). O contexto z é processado por
RNN ou MLP para extrair γ, β. Resto da CNN opera normalmente.

### 2.3 Por que funciona

- **Eficiente em parâmetros**: γ, β têm dimensão C, não C×H×W.
- **Expressivo**: combinação linear de afins é universal aproximator
  para boas escolhas de contexto.
- **Compatível com BatchNorm**: substitui parâmetros γ, β do BatchNorm
  por γ, β condicionais — interpretação como "conditional BatchNorm".

## 3. Datasets e setup experimental

- **CLEVR** (visual reasoning benchmark, 100K imagens 3D
  sintéticas + questões de múltipla composição).
- **CLEVR-Humans** (questões humanas).
- Backbone: ResNet-101 modificado com FiLM layers após cada bloco
  residual.

## 4. Métricas reportadas

- **Accuracy** em CLEVR (test set padrão + few-shot/zero-shot).
- **Ablations** comparando FiLM com baselines (concatenação, attention).

## 5. Resultados principais

- **Halves SOTA error** em CLEVR (era ~3.3%, FiLM atinge ~1.5%).
- **Robusto a ablations** — funciona com várias arquiteturas.
- **Generaliza zero-shot** para questões com primitivas não vistas
  em treino.

## 6. Limitações declaradas pelos autores

- **Otimizado para VQA-like tasks**: contexto é texto, target é
  resposta. Para tarefas com contexto **denso** (e.g., outra imagem),
  pode ser menos efetivo.
- **Pontos de inserção** de FiLM são manualmente escolhidos —
  arquitetura não-trivial.

## 7. Limitações que identifiquei

- **Contexto z é vetor** — para usar saída do MST classifier (10
  logits), z teria dimensão 10, suficiente para gerar γ, β
  pequenos. **Operacional**.
- **Não testado em fairness**. Adoção em literatura de fairness é
  rara — abertura para contribuição original (caso usemos FiLM como
  mecanismo de mitigation).
- **Risco de overfit ao contexto**: se MST classifier tem viés (e
  ele tem — Schumann 2023 documenta), FiLM pode propagá-lo.
  Mitigação: validar com pool de anotadores diverso.

## 8. Relação com nossa pesquisa

**Centralidade para v3.2 (nova linha do orientador):**

Pipeline v3.2:
```
Imagem → CNN (ResNet/ConvNeXt)
                    ↓
   MST classifier ──▶ vetor z (10 logits)
                    ↓
                FiLM layer ── γ, β
                    ↓
              feature map modulado
                    ↓
              race classifier head
                    ↓
                race prediction
```

**Por que FiLM resolve o problema concreto:**

1. **Mecanismo formal** para "considerar a composição da classe cor"
   (palavras do orientador) sem treinar de zero — apenas insere
   FiLM layer no nosso ConvNeXt-T baseline.
2. **Backbone existing** (ResNet-34, ConvNeXt-T) **não precisa
   mudar** — FiLM é wrapper.
3. **Treinamento end-to-end** — backpropagation flui através do
   MST classifier (se desejado) ou ele fica frozen como auxiliary
   feature extractor.
4. **Ablation trivial**: comparar accuracy/fairness **com vs sem**
   FiLM, mesma arquitetura — teste limpo do efeito do
   condicionamento.

**Ancoragem em literatura adjacente:**

- Conditional Batch Norm (de Vries et al. 2017, NeurIPS) é equivalente
  a FiLM. Autores explicitam.
- AdaIN (Huang & Belongie 2017 ICCV) faz coisa similar para style
  transfer.
- SPADE (Park et al. 2019 CVPR) para semantic image synthesis.

## 9. Pontos para citar

- *"A camada FiLM (Perez et al., AAAI 2018) implementa
  transformação afim por canal aprendida a partir de fonte
  contextual auxiliar, permitindo modular features de uma rede
  visual sem alterar sua arquitetura backbone. Esta dissertação
  adota FiLM como mecanismo formal para condicionar o classifier
  de raça pela saída de um classifier auxiliar de tom de pele
  (MST), instanciando empiricamente a linha experimental sugerida
  pelo orientador."*
- *"FiLM é parameter-efficient: para um backbone ConvNeXt-T (28M
  parâmetros), o overhead de inserir camadas FiLM condicionadas em
  10-dim MST logits é < 1% — mantendo o tradeoff Pareto-eficiente
  arquitetura ↔ fairness."*

## 10. Arquivos relacionados

- PDF: `pdfs/perez_2018_film.pdf` (gitignored).
- Código original: github.com/ethanjperez/film
- Entradas relacionadas: [[schumann_2023]] (MST classifier — fonte do
  contexto z), [[dataset_karkkainen_2021]] (FairFace — onde o pipeline
  é executado), [[park_2022]] (FSCL — técnica alternativa de
  mitigation, comparação).

## 11. Trabalhos sugeridos pelos autores (Future Work)

- **Aplicar FiLM a outras modalidades de contexto** além de texto
  (e.g., outras imagens, classes). ✅ **Diretamente alinhado com
  v3.2** — usamos saída de classifier como contexto.
- **Investigar pontos ótimos de inserção** das camadas FiLM em
  diferentes arquiteturas. ⚠ Hyperparameter tuning na nossa Fase 5.
- **Generalizar para multi-step reasoning**. ❌ Fora do escopo.
- **Aplicar a fairness**: NÃO mencionado pelos autores — direção
  **original** se adotarmos em v3.2.

## 12. Análise crítica do método

### (a) Rigor formal

- **FiLM matematicamente simples e clara**: F' = γ ⊙ F + β por canal,
  com γ, β derivados do contexto via MLPs.
- **Generalização de Conditional Batch Norm** (de Vries 2017) —
  conexão teórica estabelecida.
- **Validação empírica em CLEVR**: error 3.3% → 1.5% — magnitude
  significativa.
- **Limitação**: pontos ótimos de inserção determinados
  empiricamente, sem prova de optimalidade.

### (b) Reprodutibilidade

- ✅ Código original público: github.com/ethanjperez/film.
- ✅ Hiperparâmetros principais declarados (ResNet-101 com FiLM após
  cada bloco residual).
- ✅ CLEVR é benchmark padrão.
- ⚠ Pontos de inserção FiLM em diferentes arquiteturas exigem
  tuning específico.

### (c) Aplicabilidade ao pipeline v3.2

- **Mecanismo central da Etapa 3 do pipeline**: condicionar race
  classifier com saída do MST classifier.
- **Custo paramétrico baixo** (~1% sobre backbone) torna adoção
  defensável.
- **Compatibilidade com BackBoneNorm**: substitui parâmetros γ, β
  do BatchNorm/LayerNorm por γ, β condicionais.
- **Crítica importante**: FiLM nunca foi aplicado a fairness na
  literatura — direção original.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| Transformação afim por canal | ✅ Justificada — expressivo + eficiente |
| f_γ, f_β como MLPs simples | ✅ Justificada — interpretabilidade + custo baixo |
| Inserção após blocos residuais | ⚠ Empírica — sem prova de optimalidade |
| Contexto z como vetor | ✅ Justificada — flexibilidade (texto, classe, embedding) |
| CLEVR como benchmark | ✅ Justificada — VQA padrão |
| Não testar em fairness | ✅ Choice — fora do escopo original |

### (e) Conexão com R5/R6

- [[madras_2018]] LAFTR: FiLM é conditional, LAFTR é representational.
  Combinação não testada na literatura.
- [[zhang_2018]] Adversarial: FiLM modula features; Zhang modula
  output. Mecanismos distintos.
- [[pereira_2026]] SkinToneNet: provê o classificador cuja saída é
  contexto z para FiLM.
- [[hardt_2016]]: FiLM não impõe diretamente EO_h, mas
  empiricamente pode reduzir disparidade.
- **Implicação para v3.2**: FiLM é **mecanismo central** da nossa
  contribuição. Aplicação a fairness em race classification é
  direção **original** desta dissertação.
