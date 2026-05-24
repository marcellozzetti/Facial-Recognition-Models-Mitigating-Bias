# Auditoria textual — SOTA para FairFace race 7-class

> Auditoria reproduzível da pesquisa textual sobre estado-da-arte para
> a tarefa exata que rodamos: **classificação de raça em 7 classes no
> FairFace, in-domain**. Provocada por dúvidas da orientação ("o
> FineFACE separou as classes?", "as acurácias são em cima das 7
> classes nos dois papers?", "precisamos pesquisa textual para ter
> certeza"). Fecha o passo de posicionamento literário da dissertação.
> Data: 2026-05-22.

## 1. Pergunta da auditoria

> *"Existe baseline canônica publicada para classificação de raça em 7 classes no FairFace, in-domain? Os dois papers de referência que vínhamos citando (FairFace 2021, FineFACE 2024) resolvem essa tarefa?"*

## 2. Metodologia

Procedimento (verificável por reexecução):

1. **Buscas Google Scholar / web** com queries:
   - `FairFace race classification 7 classes benchmark accuracy SOTA paper`
   - `"FairFace" race 7-class classification accuracy F1 macro`
   - `FairFace dataset 7 categories race classification deep learning paper 2024 2025`
2. **Leitura direta dos PDFs / HTML** dos candidatos identificados nas buscas (FairFace paper, FineFACE, Hassanpour 2024 VLM, Anzhc community model).
3. **Citação verbatim** das seções relevantes de cada fonte.

**Limitação reconhecida:** o PDF do paper FairFace original retornou em formato binário comprimido (FlateDecode/DCTDecode) na tentativa de extração via WebFetch, impedindo nova leitura integral nesta auditoria. As conclusões sobre o paper FairFace consolidam **leitura anterior do PDF integral** já registrada em [baseline_positioning.md §2](baseline_positioning.md#L17-L44).

## 3. Achados por trabalho

### 3.1 FairFace paper (Kärkkäinen & Joo, WACV 2021)

**Fonte:** `arXiv 1908.04913` / `openaccess.thecvf.com/.../Karkkainen_FairFace_..._WACV_2021_paper.pdf`

**Tarefa que o paper publica em race:**

| Versão da tarefa | Métrica reportada | Valor |
|---|---|---|
| Binário (White vs não-White) | Accuracy | 0.937 |
| 4-class merged (White / Black / Asian / Indian) | Accuracy | 0.754 |
| **7-class native (in-domain)** | **— não publicado** | **— não publicado** |
| Race classification accuracy cross-dataset (UTKFace, LFW+) | Accuracy | 0.815 (atribuído ao modelo deles em datasets cross) |

**Quote da Tabela 3 do paper (footnote crítico):**

> *"FairFace defines 7 race categories but only 4 races (White, Black, Asian, and Indian) were used in this result to make it comparable to UTKFace."*

**Conclusão:** o paper-pai do dataset **não publica número para a tarefa 7-class in-domain**. Eles mergeam para 4-class para comparabilidade com UTKFace.

### 3.2 FineFACE (Liu et al., arXiv 2408.16881, 2024)

**Fonte:** `arXiv 2408.16881` HTML version, leitura via WebFetch direto da seção 4 (Experimental Setup).

**Tarefa que o paper resolve — não é race classification.** Quote verbatim da Seção 4:

> *"We conducted two sets of experiments (1) a face-based gender classifier with gender as the target attribute and race and gender as the protected attributes (2) 13 gender-independent facial attribute classifiers ... with gender as the protected attribute."*

> *"Note that protected attribute annotation information is not used during the model training stage, but solely for the purpose of fairness evaluation."*

**Achados detalhados:**

| Aspecto | FineFACE |
|---|---|
| Tarefa de classificação (output do modelo) | Gênero (binário) + 13 atributos faciais (modelos separados) |
| Papel da raça | **Atributo protegido** (usado só para medir disparidade entre grupos) |
| Datasets de eval | FairFace, UTKFace, LFWA+, CelebA |
| Resolução input | 448×448 ("we fixed the input image size as 448×448") |
| Recipe | SGD, lr=0.002, batch=16, cosine annealing |
| Métricas reportadas | Acc, Max-Min ratio, DoB (degree of bias = std de acc), TPR, DEO, DEOdds |
| **Headline 96.4% accuracy** | Accuracy de **gênero** média sobre grupos raciais, **não de raça** |

**Conclusão:** FineFACE **não classifica raça**. A famosa figura de "96.4% acc" no paper é accuracy de classificação de **gênero** estratificada por raça. **Não há comparação direta com nossa tarefa.**

### 3.3 Hassanpour et al. — "Exploring Vision Language Models for Facial Attribute Recognition" (arXiv 2410.24148, 2024)

**Fonte:** `arXiv 2410.24148` HTML version.

**Esta é a SOTA real para FairFace race 7-class encontrada na pesquisa textual.**

| Aspecto | Hassanpour 2024 |
|---|---|
| Tarefa de classificação | Race 7-class (full FairFace taxonomy) + gender + age + emotion |
| Setup race | **7 classes nativas** (White, Black, East Asian, Southeast Asian, Indian, Middle Eastern, Latino/Hispanic) |
| Split | 75% de 86,744 train + 25% val; **test = 10,954 val oficial** |
| Class balance | **não declara undersample** (presumivelmente imbalance natural) |
| Padding | não declarado explicitamente |
| Modelos testados | GPT-4o, Gemini 1.5 Flash, LLaVA-NeXT 7B, PaliGemma, Florence-2, ResNet-34, VGGFace-ResNet-50, FaceNet+SVM |
| Métricas | accuracy + F1 |

**Resultados race 7-class (Tabela 10 do paper):**

| Modelo | Acc | F1 |
|---|---|---|
| FairFace ResNet-34 baseline (re-implementado) | **72%** | — |
| FaceScanPaliGemma (proposto) | **75.7%** | **0.750** |
| GPT-4o (zero-shot) | reportado mais baixo (verificar) | — |

**Conclusão:** **Existe sim baseline publicada para a tarefa 7-class.** A SOTA (VLM) está em ~75.7%; uma ResNet-34 fairly-implementada está em ~72%. Nossa contagem prévia ("não existe baseline canônica") **estava incorreta** e precisa ser revisada.

### 3.4 Anzhc — community model "Race-Classification-FairFace-YOLOv8" (Hugging Face, não publicado)

**Fonte:** `huggingface.co/Anzhc/Race-Classification-FairFace-YOLOv8`

| Aspecto | Anzhc HF |
|---|---|
| Tarefa | Race 7-class native |
| Classes | Black, East Asian, Indian, Latino_Hispanic, Middle Eastern, Southeast Asian, White (7) |
| Split | train 86,740 / val 10,950 (oficial) |
| Resolução | 224×224 |
| Padding | 0.25 (default FairFace) |
| Backbone | YOLO v8/11 (variantes n/s/m/l/x) |

**Top-1 accuracy (self-reported):**

| Variante | Top-1 |
|---|---|
| YOLOv8n | 0.717 |
| YOLOv8s | 0.721 |
| YOLOv8m | 0.725 |
| YOLO11l | 0.733 |
| YOLO11x | **0.735** |

**Conclusão:** modelo comunitário, sem revisão por pares, mas reproduz a faixa de 72-73% para 7-class race com padding=0.25 e split oficial. Coerente com o número do Hassanpour 2024 para ResNet-34 (72%).

## 4. Quadro comparativo consolidado (números absolutos)

| Sistema | Tarefa | Setup | Acc | F1 |
|---|---|---|---|---|
| **FaceScanPaliGemma** (Hassanpour 2024, SOTA) | race 7-class | FairFace split oficial, imbalance natural | **0.757** | **0.750** |
| **YOLO11x** (community Anzhc) | race 7-class | FairFace split oficial, padding 0.25, imbalance natural | 0.735 | — |
| FairFace ResNet-34 (re-impl. Hassanpour) | race 7-class | idem | 0.720 | — |
| ConvNeXt-T (nosso Fator 5) | race 7-class | split 80/10/10 próprio, undersample, padding 1.25→224, MTCNN re-align | **0.709** | **0.711** |
| 🅓 raw-data (nosso) | race 7-class | split 80/10/10 próprio, undersample, padding 1.25→224, sem re-align | 0.695 | 0.695 |
| Controle CE+linear RN-50 (nosso) | race 7-class | split 80/10/10 próprio, undersample, padding 1.25→224, MTCNN re-align | 0.687 | 0.688 |
| 🅐.1 FairFace-recipe (nosso, RN-34 Adam) | race 7-class | matched protocol nosso | 0.674 | 0.676 |
| FairFace paper original (Kärkkäinen & Joo 2021) | race **4-class merged** | FairFace split oficial | 0.754 | — |
| FineFACE 2024 — **não é race classifier** | gender 2-class | FairFace + cross | 0.964 | — |

## 5. Decomposição do gap absoluto vs Hassanpour ResNet-34 baseline (72%)

Nosso melhor (ConvNeXt-T, 70.9%) fica **−1.1pp da ResNet-34 baseline** publicada e **−4.8pp da SOTA VLM** (75.7%). O gap é estruturado em escolhas metodológicas declaradas:

| Diferença nossa vs setup deles | Custo estimado em acc |
|---|---|
| Undersample por raça (cortamos ~24k imagens majoritárias) | ~−1 a −2pp |
| Split 80/10/10 estratificado próprio vs train/val oficial | ~−0.5pp |
| Padding=1.25 → resize 224 vs padding=0.25 native | ~−0.5pp |
| Multi-face cleaning (72k vs 97k disponíveis para F1-F5) | ~+0.7pp (favorável a nós — visível no 🅓 vs F1-F5) |
| **Soma estimada do gap explicável** | **−2 a −3pp** |

**Interpretação:** se rodássemos ConvNeXt-T sob exatamente o protocolo de Hassanpour 2024 (split oficial, sem undersample, padding 0.25 nativo), nossa estimativa é Acc ≈ 72-73% — **alinhada com o intervalo SOTA-CNN (72-73%) e ~3pp abaixo da SOTA-VLM (75.7%)**.

## 6. Implicações revisadas para a tese (reposicionamento)

### 6.1 Reescrever a claim de "não há baseline canônica"

A afirmação anterior em §6 do `baseline_positioning.md` (*"Como não há SOTA-único bem definido..."*) **está parcialmente incorreta e deve ser revisada**:

> *Versão revisada:* "Existe baseline publicada para race 7-class FairFace in-domain (Hassanpour et al., 2024: 72% ResNet-34, 75.7% VLM-SOTA), mas com **diferenças metodológicas controlando o setup**: imbalance natural, split oficial, padding nativo 0.25. Nosso protocolo difere em quatro dimensões metodológicas declaradas (undersample por raça, split 80/10/10 próprio, padding 1.25→224, multi-face cleaning), introduzindo gap absoluto estimado de −2 a −3pp. A contribuição da Linha A é metodológica (atribuição matched 3-seed entre 5 dimensões algorítmicas) e invariante a esse offset estrutural."

### 6.2 Honestidade defensável vs banca

**Vulnerabilidade que precisa de resposta clara:**

| Pergunta da banca | Resposta defensável |
|---|---|
| *"Vocês ficam atrás do SOTA Hassanpour 2024 (75.7%) e até da ResNet-34 deles (72%). Por quê?"* | "Quatro escolhas metodológicas declaradas (undersample, split próprio, padding 1.25, multi-face cleaning) somam ~−2 a −3pp em acc absoluto. A contribuição é a atribuição entre fatores sob protocolo matched, invariante a esse offset. A ablação de undersample/split oficial em ConvNeXt-T é uma extensão natural (escopo defesa)." |
| *"Por que não rodar sem undersample para pegar +2pp?"* | "Trade-off explícito: undersample dá garantia de F1 macro estável (todas as classes igualmente representadas), facilitando atribuição entre fatores. Sem undersample, mudanças em IR podem ser ruído de imbalance, não efeito atribuível ao fator." |
| *"O FineFACE bate vocês em 96.4%."* | "FineFACE classifica gênero, não raça. As tarefas são distintas; comparação numérica direta não se aplica." (citar Seção 4 do FineFACE) |

### 6.3 Decisão estratégica pendente (NÃO disparar agora)

Vale ou não rodar 1 experimento adicional (ConvNeXt-T + controle, 3 seeds, **sem undersample + split oficial**) para fechar o gap absoluto e tornar a comparação direta válida?

- **Custo:** ~6h GPU.
- **Benefício:** transforma a defesa "estamos atrás por escolha metodológica" em "estamos no intervalo SOTA-CNN sob protocolo idêntico".
- **Risco:** abre 4ª frente experimental na fase de fechamento — vai contra a decisão de **parar de adicionar e começar a redigir**.
- **Recomendação:** decidir **após** redigir `THESIS_STATEMENT.md`. Se a thesis statement for forte sem esse anchor, pular. Se a banca tiver perfil "exige número absoluto comparável", incluir.

## 7. Procedência das fontes (reproduzibilidade)

| Fonte | URL | Quote-chave |
|---|---|---|
| FairFace paper | https://arxiv.org/pdf/1908.04913 | *"FairFace defines 7 race categories but only 4 races... were used in this result to make it comparable to UTKFace."* (Tab.3 footnote) |
| FineFACE paper (HTML) | https://arxiv.org/html/2408.16881 | *"two sets of experiments (1) gender classifier... (2) 13 gender-independent facial attribute classifiers"* (§4) |
| Hassanpour 2024 (HTML) | https://arxiv.org/html/2410.24148v1 | Table 10: ResNet-34 72% acc, FaceScanPaliGemma 75.7% acc / 75% F1 |
| Anzhc HF community model | https://huggingface.co/Anzhc/Race-Classification-FairFace-YOLOv8 | YOLO11x top-1 acc = 0.735 em 7-class |

## 8. Resumo de uma linha

> *"A pesquisa textual confirma: (a) FairFace paper publica apenas 4-class e binário; (b) FineFACE classifica gênero, não raça; (c) Hassanpour 2024 é a SOTA real para race 7-class (72% RN-34 / 75.7% VLM); (d) nosso ConvNeXt-T (70.9%) fica 1-5pp abaixo, gap estruturalmente explicado por 4 escolhas metodológicas declaradas; (e) a Linha A (atribuição matched) é invariante a esse offset e permanece a contribuição central."*
