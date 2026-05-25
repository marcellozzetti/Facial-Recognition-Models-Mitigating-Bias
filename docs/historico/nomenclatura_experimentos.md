# Nomenclatura dos experimentos — vocabulário padronizado

> Documento de padronização. Substitui os códigos enigmáticos
> (🅐.1, 🅐.2, 🅓, 🅑, 🅔, Teste A/B/C1/D, Combo #1-4) por termos
> compreensíveis em linguagem natural. Adotado a partir de 2026-05-24.
> Docs históricos preservam os códigos antigos (registro temporal);
> todo material novo (incluindo a redação dos capítulos da dissertação
> e a apresentação ao orientador) usa esta nomenclatura.

## 1. Princípio de nomenclatura

Cada experimento tem **dois identificadores**:

1. **Nome descritivo** (usado no corpo do texto, slides, conversas)
2. **Código curto** (usado em tabelas, gráficos, paths de arquivos)

O código curto NÃO usa emoji ou letras enigmáticas — usa sigla
mnemônica derivada do propósito do experimento.

## 2. Taxonomia completa (4 categorias)

### 2.1 Categoria A — Cinco Fatores de Atribuição Causal (Linha A — principal)

Estes JÁ tinham nomenclatura clara, mantidos sem alteração:

| Código | Nome descritivo | Variável manipulada |
|---|---|---|
| F1 | Fator 1 — Conjunto de dados (limpeza) | multi-face cleaning vs raw |
| F2 | Fator 2 — Topologia da camada de saída | MLP vs linear |
| F3 | Fator 3 — Família de função de custo | CE, ArcFace, AdaFace, MagFace |
| F4 | Fator 4 — Paradigma de aprendizado | CE+linear vs CE+SupCon |
| F5 | Fator 5 — Rede dorsal pré-treinada | ResNet-50, RN-34, ViT-B/16, ConvNeXt-T |

### 2.2 Categoria B — Experimentos Adicionais de Posicionamento

Substituem os "anchors" 🅐.1, 🅐.2, 🅓, 🅔.

| Código | Nome descritivo (novo) | Código antigo (deprecated) | O que isola |
|---|---|---|---|
| **Exp-FairFace** | Reprodução do recipe do paper FairFace original | 🅐.1 | Recipe do paper-pai (RN-34 + Adam lr=1e-4) |
| **Exp-FineFACE** | Reprodução do recipe do paper FineFACE (sem multi-expert) | 🅐.2 | Recipe do paper de fairness 2024 sem a arquitetura especializada |
| **Exp-DadosBrutos** | Avaliação com dados brutos (sem nosso pré-processamento) | 🅓 | Efeito do nosso multi-face cleaning + MTCNN re-alignment |
| **Exp-ProtocoloSOTA** | Reprodução integral do protocolo SOTA (AlDahoul et al. 2024) | 🅔 | Padding 0.25 + split oficial + sem subamostragem |

### 2.3 Categoria C — Análise de Robustez

Substitui a ablação 🅑.

| Código | Nome descritivo (novo) | Código antigo | Hipótese testada |
|---|---|---|---|
| **Rob-SemSubamostragem** | Análise de robustez sem subamostragem por classe | 🅑 | Alavanca ConvNeXt-T persiste sem class balance? |

### 2.4 Categoria D — Auditoria Empírica de Hiperparâmetros

Substituem os "Testes A/B/C1/D".

| Código | Nome descritivo (novo) | Código antigo | Hiperparâmetro testado |
|---|---|---|---|
| **Aud-Paciência** | Auditoria da paciência da parada antecipada | Teste A | early_stopping_patience: 5 → 15 |
| **Aud-Dropout** | Auditoria do dropout nas características | Teste C1 | model.dropout: 0.2 → 0.0 |
| **Aud-Augmentation** | Auditoria de aumento de dados moderno | Teste B | train_use_trivialaugment: false → true |
| **Aud-Oversampling** | Auditoria de balanceamento por oversampling | Teste D | data.balance: none → oversample |

### 2.5 Categoria E — Análises Pós-Treinamento (Combo defesa-fechamento)

Substituem "Combo #1-#4".

| Código | Nome descritivo (novo) | Código antigo | Técnica científica |
|---|---|---|---|
| **PT-Interseccional** | Análise de equidade interseccional (raça × gênero × idade) | Combo #1 | Buolamwini & Gebru FAccT 2018 |
| **PT-Ensemble** | Agregação por ensemble de sementes (deep ensemble) | Combo #2 | Lakshminarayanan NeurIPS 2017 |
| **PT-TTA** | Test-Time Augmentation com transformações geometricamente seguras | Combo #3 | Krizhevsky 2012 (10-crop) |
| **PT-Calibração** | Calibração via temperature scaling + threshold per-class | Combo #4 | Guo et al. ICML 2017 |

## 3. Sequência completa dos experimentos (ordem narrativa)

Para apresentação ao orientador e na redação da tese:

```
Fase 1 — Atribuição causal (5 Fatores)
  F1 → F2 → F3 → F4 → F5

Fase 2 — Posicionamento absoluto (4 Experimentos Adicionais)
  Exp-FairFace
  Exp-FineFACE
  Exp-DadosBrutos
  Exp-ProtocoloSOTA

Fase 3 — Robustez do achado central (1 Análise)
  Rob-SemSubamostragem

Fase 4 — Auditoria empírica de código (4 Auditorias)
  Aud-Paciência
  Aud-Dropout
  Aud-Augmentation
  Aud-Oversampling

Fase 5 — Análises pós-treinamento para confiabilidade (4 Análises)
  PT-Interseccional
  PT-Ensemble
  PT-TTA
  PT-Calibração
```

**Total: 5 + 4 + 1 + 4 + 4 = 18 experimentos/análises** distribuídos
em **5 fases** com propósitos distintos.

## 4. Glossário de termos técnicos relacionados (Haykin-aligned)

| Termo nosso | Equivalente Haykin |
|---|---|
| Rede dorsal pré-treinada | rede neural pré-treinada com aprendizado por transferência |
| Camada de saída | camada classificadora final |
| Vetor de características | vetor de embeddings (Haykin §2.10) |
| Função de custo | função de erro / função objetivo (Haykin §3.6) |
| Razão de disparidade | razão entre desempenhos máximo e mínimo por grupo |
| Lote | conjunto de amostras processado em um forward pass |
| Taxa de aprendizagem | parâmetro de passo do gradiente descendente |
| Parada antecipada | critério de interrupção do treinamento por estagnação |
| Subamostragem | amostragem de subconjunto da classe majoritária |
| Sobreamostragem (oversampling) | replicação de amostras da classe minoritária |

## 5. Diretriz de uso

### Para a redação dos capítulos da dissertação (a partir de agora):

**SEMPRE usar nomes descritivos** ("Reprodução do recipe FairFace
original", "Análise de robustez sem subamostragem", etc.) no corpo do
texto. O código curto pode aparecer em parênteses na primeira menção:

> *"O experimento de reprodução do recipe FairFace (Exp-FairFace)
> entrega F1=0.676 ± 0.006..."*

### Para tabelas e gráficos:

Usar o **código curto** para economia de espaço, com legenda
explicativa abaixo se necessário.

### Para a apresentação ao orientador:

Usar EXCLUSIVAMENTE nomes descritivos. Nada de emoji ou códigos
enigmáticos. Cada slide pode listar 1 código curto entre parênteses
na nota de rodapé.

### Para docs históricos (já escritos):

**NÃO alterar.** Os emoji 🅐.1, 🅐.2, 🅓, 🅑, 🅔 e os Testes A/B/C1/D
são registro temporal e ficam preservados nos docs originais. Esta
padronização vale para material NOVO a partir de 2026-05-24.

## 6. Tabela-resumo (consulta rápida)

| Categoria | Quantidade | Códigos descritivos |
|---|---|---|
| Fatores de atribuição causal | 5 | F1, F2, F3, F4, F5 |
| Experimentos de posicionamento | 4 | Exp-FairFace, Exp-FineFACE, Exp-DadosBrutos, Exp-ProtocoloSOTA |
| Análise de robustez | 1 | Rob-SemSubamostragem |
| Auditoria de hiperparâmetros | 4 | Aud-Paciência, Aud-Dropout, Aud-Augmentation, Aud-Oversampling |
| Análises pós-treinamento | 4 | PT-Interseccional, PT-Ensemble, PT-TTA, PT-Calibração |
| **TOTAL** | **18** | — |
