# Auditoria Semântica da Literatura — Capítulo (rascunho)

> **Documento gerado por** `scripts/semantic_search_corpus.py`.
> Material destinado à escrita da dissertação (Diretriz 8). Reproduzível: re-rodar o script regenera este arquivo.

**Data de geração:** 2026-05-15  
**Modelo de embedding:** `BAAI/bge-small-en-v1.5` (sentence-transformers)  
**Métrica de similaridade:** cosseno sobre embeddings L2-normalizados  
**Top-K por query:** 15

---

## 1. Metodologia

Esta auditoria semântica complementa a revisão sistemática ([sota_review.md](sota_review.md)). Enquanto aquela usa busca booleana em venues, esta usa **similaridade semântica densa** sobre o corpus completo de papers que citam o FairFace + as referências do próprio FairFace, para mitigar o viés de palavra-chave da busca booleana.

**Protocolo:**

1. Corpus: `docs/literature_corpus.csv` (555 documentos; 374 com abstract).

2. Representação: `título + abstract` codificado em embedding denso.

3. Consultas: 14 *research queries* — 10 alinhadas às diretrizes do orientador, 4 sondas de **novidade/overlap** (Q11–Q14) que testam diretamente se o delta da tese já existe.

4. Ranqueamento: top-K por similaridade de cosseno.

5. Análise: leitura dos top-K das sondas de novidade para confirmar/refutar overlap.


**Ameaças à validade declaradas:** (i) corpus OpenAlex pode não ser exaustivo vs Google Scholar (479 vs ~900 brutos — diferença por deduplicação); (ii) similaridade semântica captura proximidade temática, não prova ausência de overlap — leitura humana dos top-K é mandatória antes de afirmar novidade; (iii) abstracts ausentes degradam o sinal para alguns docs.

---

## 2. Caracterização do corpus

| Fonte | Qtd |
|---|---:|
| `fairface-citing` | 479 |
| `fairface-ref` | 76 |

**Distribuição temporal (papers que citam):**

| Ano | Qtd |
|---|---:|
| 2021 | 41 |
| 2022 | 59 |
| 2023 | 109 |
| 2024 | 120 |
| 2025 | 128 |
| 2026 | 22 |

---

## 3. Resultados por query

### Q01_mlp_head (diretriz-2)

> *Query:* Multi-layer perceptron classification head topology for race classification on FairFace

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.805 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 2 | 0.802 | 2015 | FaceNet: A unified embedding for face recognition and clustering | fairface-ref |
| 3 | 0.801 | 2024 | Hybrid deep Ensemble for Fine-Grained Race Estimation | fairface-citing |
| 4 | 0.792 | 2016 | The MegaFace benchmark: 1 million faces for recognition at scale | fairface-ref |
| 5 | 0.791 | 2017 | InclusiveFaceNet: Improving face attribute detection with race and gender diversity | fairface-ref |
| 6 | 0.784 | 2024 | Surveying Racial Bias in Facial Recognition: Balancing Datasets and Algorithmic Enhancements | fairface-citing |
| 7 | 0.782 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 8 | 0.781 | 2024 | Study on the Generation and Comparative Analysis of Ethnically Diverse Faces for Developing a Multiracial Face | fairface-citing |
| 9 | 0.780 | 2015 | Deep face recognition (VGGFace) | fairface-ref |
| 10 | 0.780 | 2023 | F3: Fair and Federated Face Attribute Classification with Heterogeneous Data | fairface-citing |
| 11 | 0.777 | 2024 | SA-SVD: Mitigating Bias in Face Recognition by Fair Representation Learning | fairface-citing |
| 12 | 0.774 | 2023 | On bias and fairness in deep learning-based facial analysis | fairface-citing |
| 13 | 0.771 | 2013 | Learning fair representations | fairface-ref |
| 14 | 0.769 | 2025 | Bias Mitigation Strategies for Facial Attribute Classification Leveraging Fine-grained Features and ChatGPT | fairface-citing |
| 15 | 0.769 | 2021 | Fairness Testing of Deep Image Classification with Adequacy Metrics | fairface-citing |

### Q02_optuna_hpo (diretriz-3)

> *Query:* Optuna hyperparameter optimization for face recognition demographic fairness

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.885 | 2026 | Assessing Demographic Bias and Fairness in Facial Recognition Systems: A Framework | fairface-citing |
| 2 | 0.837 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 3 | 0.833 | 2023 | Uncovering Bias in the Face Processing Pipeline: An Analysis of Popular and State-of-the-Art Algorithms Across | fairface-citing |
| 4 | 0.832 | 2022 | Enhancing Fairness in Face Detection in Computer Vision Systems by Demographic Bias Mitigation | fairface-citing |
| 5 | 0.818 | 2024 | Fine-Grained Rebalancing of Datasets for Correct Demographic Classification | fairface-citing |
| 6 | 0.816 | 2023 | Leveraging Diffusion and Flow Matching Models for Demographic Bias Mitigation of Facial Attribute Classifiers | fairface-citing |
| 7 | 0.812 | 2023 | On Adversarial Robustness of Demographic Fairness in Face Attribute Recognition | fairface-citing |
| 8 | 0.812 | 2024 | FaDE: A Face Segment Driven Identity Anonymization Framework For Fair Face Recognition | fairface-citing |
| 9 | 0.811 | 2016 | The MegaFace benchmark: 1 million faces for recognition at scale | fairface-ref |
| 10 | 0.811 | 2023 | Balancing Biases and Preserving Privacy on Balanced Faces in the Wild | fairface-citing |
| 11 | 0.810 | 2025 | Leveraging diffusion and Flow Matching Models for demographic bias mitigation of facial attribute classifiers | fairface-citing |
| 12 | 0.809 | 2025 | Fairness without Labels: Pseudo-Balancing for Bias Mitigation in Face Gender Classification | fairface-citing |
| 13 | 0.806 | 2023 | On bias and fairness in deep learning-based facial analysis | fairface-citing |
| 14 | 0.806 | 2024 | Demographic bias mitigation at test-time using uncertainty estimation and human–machine partnership | fairface-citing |
| 15 | 0.804 | 2015 | FaceNet: A unified embedding for face recognition and clustering | fairface-ref |

### Q03_supcon (diretriz-5)

> *Query:* Supervised contrastive learning SupCon for demographic equity in face attribute classification

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.879 | 2025 | Leveraging diffusion and Flow Matching Models for demographic bias mitigation of facial attribute classifiers | fairface-citing |
| 2 | 0.855 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 3 | 0.850 | 2024 | A Self-Supervised Learning Pipeline for Demographically Fair Facial Attribute Classification | fairface-citing |
| 4 | 0.844 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 5 | 0.844 | 2026 | Assessing Demographic Bias and Fairness in Facial Recognition Systems: A Framework | fairface-citing |
| 6 | 0.839 | 2017 | InclusiveFaceNet: Improving face attribute detection with race and gender diversity | fairface-ref |
| 7 | 0.838 | 2023 | Leveraging Diffusion and Flow Matching Models for Demographic Bias Mitigation of Facial Attribute Classifiers | fairface-citing |
| 8 | 0.834 | 2023 | CAT: Controllable Attribute Translation for Fair Facial Attribute Classification | fairface-citing |
| 9 | 0.832 | 2025 | Bias Mitigation Strategies for Facial Attribute Classification Leveraging Fine-grained Features and ChatGPT | fairface-citing |
| 10 | 0.827 | 2025 | Improving Bias in Facial Attribute Classification: A Combined Impact of KL Divergence-Induced Loss Function an | fairface-citing |
| 11 | 0.821 | 2025 | Analyzing and mitigating bias of facial attribute classifiers using ChatGPT | fairface-citing |
| 12 | 0.818 | 2023 | Uncovering Bias in the Face Processing Pipeline: An Analysis of Popular and State-of-the-Art Algorithms Across | fairface-citing |
| 13 | 0.817 | 2024 | Demographic bias mitigation at test-time using uncertainty estimation and human–machine partnership | fairface-citing |
| 14 | 0.815 | 2023 | F3: Fair and Federated Face Attribute Classification with Heterogeneous Data | fairface-citing |
| 15 | 0.814 | 2017 | Face aging with conditional generative adversarial networks | fairface-ref |

### Q04_simclr (diretriz-5)

> *Query:* SimCLR self-supervised pretraining for fair face recognition

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.846 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 2 | 0.835 | 2025 | MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition | fairface-citing |
| 3 | 0.829 | 2023 | On bias and fairness in deep learning-based facial analysis | fairface-citing |
| 4 | 0.825 | 2024 | A Self-Supervised Learning Pipeline for Demographically Fair Facial Attribute Classification | fairface-citing |
| 5 | 0.814 | 2013 | Hybrid deep learning for face verification | fairface-ref |
| 6 | 0.813 | 2026 | Assessing Demographic Bias and Fairness in Facial Recognition Systems: A Framework | fairface-citing |
| 7 | 0.812 | 2024 | FaDE: A Face Segment Driven Identity Anonymization Framework For Fair Face Recognition | fairface-citing |
| 8 | 0.811 | 2022 | Real-time self-supervised achromatic face colorization | fairface-citing |
| 9 | 0.811 | 2023 | F3: Fair and Federated Face Attribute Classification with Heterogeneous Data | fairface-citing |
| 10 | 0.809 | 2026 | FairMoE: Decoupled Expert Learning for Unbiased Customized Face Generation | fairface-citing |
| 11 | 0.808 | 2016 | OpenFace: A general-purpose face recognition library with mobile applications | fairface-ref |
| 12 | 0.806 | 2015 | FaceNet: A unified embedding for face recognition and clustering | fairface-ref |
| 13 | 0.805 | 2024 | An Unconstrained Dataset for Face Recognition Across Distance, Pose, and Resolution | fairface-citing |
| 14 | 0.805 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 15 | 0.804 | 2025 | Fairness is in the Details : Face Dataset Auditing | fairface-citing |

### Q05_clip_face (diretriz-5)

> *Query:* CLIP contrastive language-image pretraining for face attribute fairness

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.856 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 2 | 0.844 | 2023 | CAT: Controllable Attribute Translation for Fair Facial Attribute Classification | fairface-citing |
| 3 | 0.843 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 4 | 0.843 | 2023 | Enhancing Fairness of Visual Attribute Predictors | fairface-citing |
| 5 | 0.831 | 2017 | InclusiveFaceNet: Improving face attribute detection with race and gender diversity | fairface-ref |
| 6 | 0.827 | 2024 | Improving Fairness using Vision-Language Driven Image Augmentation | fairface-citing |
| 7 | 0.815 | 2017 | Arbitrary facial attribute editing: Only change what you want | fairface-ref |
| 8 | 0.811 | 2023 | On bias and fairness in deep learning-based facial analysis | fairface-citing |
| 9 | 0.805 | 2026 | FairMoE: Decoupled Expert Learning for Unbiased Customized Face Generation | fairface-citing |
| 10 | 0.803 | 2023 | F3: Fair and Federated Face Attribute Classification with Heterogeneous Data | fairface-citing |
| 11 | 0.801 | 2018 | Heterogeneous face attribute estimation: A deep multi-task learning approach | fairface-ref |
| 12 | 0.800 | 2025 | MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition | fairface-citing |
| 13 | 0.800 | 2025 | Bias Mitigation Strategies for Facial Attribute Classification Leveraging Fine-grained Features and ChatGPT | fairface-citing |
| 14 | 0.797 | 2023 | DeAR: Debiasing Vision-Language Models with Additive Residuals | fairface-citing |
| 15 | 0.797 | 2025 | Leveraging diffusion and Flow Matching Models for demographic bias mitigation of facial attribute classifiers | fairface-citing |

### Q06_adaface_magface (diretriz-6)

> *Query:* AdaFace MagFace quality-adaptive margin loss for racial fairness

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.813 | 2023 | Enhancing Fairness of Visual Attribute Predictors | fairface-citing |
| 2 | 0.812 | 2017 | InclusiveFaceNet: Improving face attribute detection with race and gender diversity | fairface-ref |
| 3 | 0.811 | 2026 | Assessing Demographic Bias and Fairness in Facial Recognition Systems: A Framework | fairface-citing |
| 4 | 0.809 | 2025 | Fairness is in the Details : Face Dataset Auditing | fairface-citing |
| 5 | 0.797 | 2017 | Fairness constraints: Mechanisms for fair classification | fairface-ref |
| 6 | 0.796 | 2024 | Leveraging Diffusion Perturbations for Measuring Fairness in Computer Vision | fairface-citing |
| 7 | 0.796 | 2023 | Uncovering Bias in the Face Processing Pipeline: An Analysis of Popular and State-of-the-Art Algorithms Across | fairface-citing |
| 8 | 0.794 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 9 | 0.793 | 2022 | Enhancing Fairness in Face Detection in Computer Vision Systems by Demographic Bias Mitigation | fairface-citing |
| 10 | 0.793 | 2023 | Racial Bias in the Beautyverse: Evaluation of Augmented-Reality Beauty Filters | fairface-citing |
| 11 | 0.793 | 2025 | Saliency-based metric and FaceKeepOriginalAugment: a novel approach for enhancing fairness and Diversity | fairface-citing |
| 12 | 0.792 | 2024 | Benchmarking the Fairness of Image Upsampling Methods | fairface-citing |
| 13 | 0.792 | 2024 | Benchmarking the Fairness of Image Upsampling Methods | fairface-citing |
| 14 | 0.792 | 2024 | Surveying Racial Bias in Facial Recognition: Balancing Datasets and Algorithmic Enhancements | fairface-citing |
| 15 | 0.789 | 2025 | Algorithmic Bias Detection: A Focus on Skin Tone and Gender Fairness in AI Models | fairface-citing |

### Q07_multiface_clean (diretriz-4)

> *Query:* Multi-face image label noise filtering and dataset cleaning for FairFace

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.844 | 2014 | A data-driven approach to cleaning large face datasets | fairface-ref |
| 2 | 0.811 | 2025 | Fairness is in the Details : Face Dataset Auditing | fairface-citing |
| 3 | 0.807 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 4 | 0.801 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 5 | 0.793 | 2024 | GAMMA-FACE: GAussian Mixture Models Amend Diffusion Models for Bias Mitigation in Face Images | fairface-citing |
| 6 | 0.793 | 2025 | TONO: A Synthetic Dataset for Face Image Compliance to ISO/ICAO Standard | fairface-citing |
| 7 | 0.792 | 2024 | An Unconstrained Dataset for Face Recognition Across Distance, Pose, and Resolution | fairface-citing |
| 8 | 0.791 | 2015 | Deep face recognition (VGGFace) | fairface-ref |
| 9 | 0.789 | 2023 | CAT: Controllable Attribute Translation for Fair Facial Attribute Classification | fairface-citing |
| 10 | 0.789 | 2016 | OpenFace: A general-purpose face recognition library with mobile applications | fairface-ref |
| 11 | 0.786 | 2016 | The MegaFace benchmark: 1 million faces for recognition at scale | fairface-ref |
| 12 | 0.781 | 2025 | MST-KD: Multiple Specialized Teachers Knowledge Distillation for Fair Face Recognition | fairface-citing |
| 13 | 0.780 | 2023 | On bias and fairness in deep learning-based facial analysis | fairface-citing |
| 14 | 0.779 | 2011 | SCface - surveillance cameras face database | fairface-ref |
| 15 | 0.778 | 2025 | FaiResGAN: Fair and robust blind face restoration with biometrics preservation | fairface-citing |

### Q08_inequity_rate (metric)

> *Query:* Inequity rate Gini coefficient demographic disparity metric face recognition

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.844 | 2026 | Assessing Demographic Bias and Fairness in Facial Recognition Systems: A Framework | fairface-citing |
| 2 | 0.840 | 2022 | Enhancing Fairness in Face Detection in Computer Vision Systems by Demographic Bias Mitigation | fairface-citing |
| 3 | 0.840 | 2023 | Evaluation of targeted dataset collection on racial equity in face recognition | fairface-citing |
| 4 | 0.831 | 2023 | The Impact of Racial Distribution in Training Data on Face Recognition Bias: A Closer Look | fairface-citing |
| 5 | 0.829 | 2024 | Study on the Generation and Comparative Analysis of Ethnically Diverse Faces for Developing a Multiracial Face | fairface-citing |
| 6 | 0.828 | 2024 | Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition | fairface-citing |
| 7 | 0.828 | 2023 | Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition | fairface-citing |
| 8 | 0.826 | 2023 | FACET: Fairness in Computer Vision Evaluation Benchmark | fairface-citing |
| 9 | 0.824 | 2023 | Toward responsible face datasets: modeling the distribution of a disentangled latent space for sampling face i | fairface-citing |
| 10 | 0.821 | 2021 | Information-Theoretic Bias Assessment Of Learned Representations Of Pretrained Face Recognition | fairface-citing |
| 11 | 0.820 | 2021 | Evaluation of Gender Bias in Facial Recognition with Traditional Machine Learning Algorithms | fairface-citing |
| 12 | 0.820 | 2024 | SA-SVD: Mitigating Bias in Face Recognition by Fair Representation Learning | fairface-citing |
| 13 | 0.819 | 2024 | Surveying Racial Bias in Facial Recognition: Balancing Datasets and Algorithmic Enhancements | fairface-citing |
| 14 | 0.817 | 2017 | InclusiveFaceNet: Improving face attribute detection with race and gender diversity | fairface-ref |
| 15 | 0.816 | 2025 | Assessing bias and computational efficiency in vision transformers using early exits | fairface-citing |

### Q09_undersampling_limits (premise)

> *Query:* Limitations of class balancing and undersampling for demographic fairness in deep learning

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.842 | 2024 | Fine-Grained Rebalancing of Datasets for Correct Demographic Classification | fairface-citing |
| 2 | 0.834 | 2023 | A Fair Generative Model Using LeCam Divergence | fairface-citing |
| 3 | 0.833 | 2023 | On Adversarial Robustness of Demographic Fairness in Face Attribute Recognition | fairface-citing |
| 4 | 0.828 | 2026 | Assessing Demographic Bias and Fairness in Facial Recognition Systems: A Framework | fairface-citing |
| 5 | 0.828 | 2025 | Bias Mitigation in Deep Learning Models for Facial Recognition | fairface-citing |
| 6 | 0.823 | 2024 | Utility-Fairness Trade-Offs and how to Find Them | fairface-citing |
| 7 | 0.822 | 2024 | FairIF: Boosting Fairness in Deep Learning via Influence Functions with Validation Set Sensitive Attributes | fairface-citing |
| 8 | 0.819 | 2021 | Fairness Testing of Deep Image Classification with Adequacy Metrics | fairface-citing |
| 9 | 0.819 | 2023 | Leveraging Diffusion and Flow Matching Models for Demographic Bias Mitigation of Facial Attribute Classifiers | fairface-citing |
| 10 | 0.818 | 2025 | Fairness in Machine Learning: A Review for Statisticians | fairface-citing |
| 11 | 0.817 | 2025 | Computational fairness in adaptive neural networks | fairface-citing |
| 12 | 0.815 | 2025 | Fairness without Labels: Pseudo-Balancing for Bias Mitigation in Face Gender Classification | fairface-citing |
| 13 | 0.814 | 2022 | Enhancing Fairness in Face Detection in Computer Vision Systems by Demographic Bias Mitigation | fairface-citing |
| 14 | 0.813 | 2024 | Demographic bias mitigation at test-time using uncertainty estimation and human–machine partnership | fairface-citing |
| 15 | 0.812 | 2024 | DiversiNet: Mitigating Bias in Deep Classification Networks across Sensitive Attributes through Diffusion-Gene | fairface-citing |

### Q10_backbone (axis-backbone)

> *Query:* Vision transformer ConvNeXt backbone for fair facial attribute classification

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.878 | 2023 | CAT: Controllable Attribute Translation for Fair Facial Attribute Classification | fairface-citing |
| 2 | 0.877 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 3 | 0.847 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 4 | 0.846 | 2023 | On bias and fairness in deep learning-based facial analysis | fairface-citing |
| 5 | 0.833 | 2023 | Enhancing Fairness of Visual Attribute Predictors | fairface-citing |
| 6 | 0.825 | 2023 | F3: Fair and Federated Face Attribute Classification with Heterogeneous Data | fairface-citing |
| 7 | 0.822 | 2025 | Analyzing and mitigating bias of facial attribute classifiers using ChatGPT | fairface-citing |
| 8 | 0.817 | 2011 | Describable visual attributes for face verification and image search | fairface-ref |
| 9 | 0.816 | 2025 | Improving Bias in Facial Attribute Classification: A Combined Impact of KL Divergence-Induced Loss Function an | fairface-citing |
| 10 | 0.816 | 2025 | Bias Mitigation Strategies for Facial Attribute Classification Leveraging Fine-grained Features and ChatGPT | fairface-citing |
| 11 | 0.813 | 2017 | Arbitrary facial attribute editing: Only change what you want | fairface-ref |
| 12 | 0.809 | 2025 | Privacy-preserving face attribute classification via differential privacy | fairface-citing |
| 13 | 0.806 | 2024 | A Self-Supervised Learning Pipeline for Demographically Fair Facial Attribute Classification | fairface-citing |
| 14 | 0.806 | 2026 | SHaSaM: Submodular Hard Sample Mining for Fair Facial Attribute Recognition | fairface-citing |
| 15 | 0.805 | 2025 | Leveraging diffusion and Flow Matching Models for demographic bias mitigation of facial attribute classifiers | fairface-citing |

### Q11_moo_arch_fairness (novelty-risk)

> *Query:* Multi-objective architecture search Pareto front accuracy fairness trade-off face attribute classification

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.873 | 2022 | FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification | fairface-citing |
| 2 | 0.848 | 2024 | FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features | fairface-citing |
| 3 | 0.842 | 2023 | F3: Fair and Federated Face Attribute Classification with Heterogeneous Data | fairface-citing |
| 4 | 0.838 | 2023 | CAT: Controllable Attribute Translation for Fair Facial Attribute Classification | fairface-citing |
| 5 | 0.820 | 2025 | Privacy-preserving face attribute classification via differential privacy | fairface-citing |
| 6 | 0.819 | 2025 | Bias Mitigation Strategies for Facial Attribute Classification Leveraging Fine-grained Features and ChatGPT | fairface-citing |
| 7 | 0.814 | 2018 | Heterogeneous face attribute estimation: A deep multi-task learning approach | fairface-ref |
| 8 | 0.811 | 2026 | SHaSaM: Submodular Hard Sample Mining for Fair Facial Attribute Recognition | fairface-citing |
| 9 | 0.804 | 2017 | InclusiveFaceNet: Improving face attribute detection with race and gender diversity | fairface-ref |
| 10 | 0.803 | 2023 | On Adversarial Robustness of Demographic Fairness in Face Attribute Recognition | fairface-citing |
| 11 | 0.799 | 2017 | Fairness constraints: Mechanisms for fair classification | fairface-ref |
| 12 | 0.799 | 2011 | Describable visual attributes for face verification and image search | fairface-ref |
| 13 | 0.795 | 2025 | Fairness is in the Details : Face Dataset Auditing | fairface-citing |
| 14 | 0.795 | 2025 | Leveraging diffusion and Flow Matching Models for demographic bias mitigation of facial attribute classifiers | fairface-citing |
| 15 | 0.794 | 2016 | OpenFace: A general-purpose face recognition library with mobile applications | fairface-ref |

### Q12_pareto_epoch (novelty-core)

> *Query:* Pareto-aware epoch selection criterion within multi-objective hyperparameter optimization trial for fairness

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.795 | 2017 | Fairness constraints: Mechanisms for fair classification | fairface-ref |
| 2 | 0.795 | 2025 | Fairness in Machine Learning: A Review for Statisticians | fairface-citing |
| 3 | 0.781 | 2025 | Computational fairness in adaptive neural networks | fairface-citing |
| 4 | 0.780 | 2024 | Utility-Fairness Trade-Offs and how to Find Them | fairface-citing |
| 5 | 0.772 | 2017 | Algorithmic decision making and the cost of fairness | fairface-ref |
| 6 | 0.769 | 2024 | Demographic Bias Mitigation at Test-Time Using Uncertainty Estimation and Human-Machine Partnership | fairface-citing |
| 7 | 0.769 | 2023 | FairCaipi: A Combination of Explanatory Interactive and Fair Machine Learning for Human and Machine Bias Reduc | fairface-citing |
| 8 | 0.766 | 2024 | Fairness Testing: A Comprehensive Survey and Analysis of Trends | fairface-citing |
| 9 | 0.761 | 2024 | FairIF: Boosting Fairness in Deep Learning via Influence Functions with Validation Set Sensitive Attributes | fairface-citing |
| 10 | 0.759 | 2021 | Fairness Testing of Deep Image Classification with Adequacy Metrics | fairface-citing |
| 11 | 0.756 | 2023 | Fair Robust Active Learning by Joint Inconsistency | fairface-citing |
| 12 | 0.754 | 2025 | Algorithmic bias, fairness, and inclusivity: a multilevel framework for justice-oriented AI | fairface-citing |
| 13 | 0.754 | 2016 | Equality of opportunity in supervised learning | fairface-ref |
| 14 | 0.751 | 2024 | Fairness-Sensitive Policy-Gradient Reinforcement Learning for Reducing Bias in Robotic Assistance | fairface-citing |
| 15 | 0.751 | 2024 | Fine-Tuning a Biased Model for Improving Fairness | fairface-citing |

### Q13_decomposition (novelty-core)

> *Query:* Controlled experimental decomposition isolating dataset quality versus model architecture contribution to demographic bias

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.870 | 2024 | Dsap: Analyzing Bias Through Demographic Comparison of Datasets | fairface-citing |
| 2 | 0.841 | 2024 | Demographic Bias Mitigation at Test-Time Using Uncertainty Estimation and Human-Machine Partnership | fairface-citing |
| 3 | 0.839 | 2024 | Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition | fairface-citing |
| 4 | 0.839 | 2023 | Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition | fairface-citing |
| 5 | 0.821 | 2024 | Demographic bias mitigation at test-time using uncertainty estimation and human–machine partnership | fairface-citing |
| 6 | 0.818 | 2011 | Unbiased look at dataset bias | fairface-ref |
| 7 | 0.813 | 2023 | Simplicity Bias Leads to Amplified Performance Disparities | fairface-citing |
| 8 | 0.805 | 2024 | Fairness Feedback Loops: Training on Synthetic Data Amplifies Bias | fairface-citing |
| 9 | 0.803 | 2026 | Evaluation of Data Quality Disparity and Implications for Fair Machine Learning | fairface-citing |
| 10 | 0.801 | 2024 | DiversiNet: Mitigating Bias in Deep Classification Networks across Sensitive Attributes through Diffusion-Gene | fairface-citing |
| 11 | 0.798 | 2023 | A Fair Generative Model Using LeCam Divergence | fairface-citing |
| 12 | 0.797 | 2026 | Bias-Free? An Empirical Study on Ethnicity, Gender, and Age Fairness in Deepfake Detection | fairface-citing |
| 13 | 0.795 | 2023 | Representation Bias in Data: A Survey on Identification and Resolution Techniques | fairface-citing |
| 14 | 0.794 | 2018 | Mitigating bias in gender, age and ethnicity classification | fairface-ref |
| 15 | 0.794 | 2025 | Unbiased-Diff: Analyzing and Mitigating Biases in Diffusion Model-Based Face Image Generation | fairface-citing |

### Q14_recipe_dependent_clean (novelty-empirical)

> *Query:* Recipe-dependent effect of dataset cleaning on margin-based versus softmax-based losses for fairness

| # | sim | Ano | Título | Fonte |
|---:|---:|---:|---|---|
| 1 | 0.834 | 2025 | Fairness in Machine Learning: A Review for Statisticians | fairface-citing |
| 2 | 0.813 | 2025 | Fairness is in the Details : Face Dataset Auditing | fairface-citing |
| 3 | 0.803 | 2017 | Fairness constraints: Mechanisms for fair classification | fairface-ref |
| 4 | 0.797 | 2017 | Algorithmic decision making and the cost of fairness | fairface-ref |
| 5 | 0.793 | 2024 | Utility-Fairness Trade-Offs and how to Find Them | fairface-citing |
| 6 | 0.793 | 2011 | Unbiased look at dataset bias | fairface-ref |
| 7 | 0.791 | 2022 | Fair Representation: Guaranteeing Approximate Multiple Group Fairness for Unknown Tasks | fairface-citing |
| 8 | 0.788 | 2024 | Mutual Information-Based Fair Active Learning | fairface-citing |
| 9 | 0.787 | 2024 | Exploring Fairness-Accuracy Trade-Offs in Binary Classification: A Comparative Analysis Using Modified Loss Fu | fairface-citing |
| 10 | 0.787 | 2026 | Evaluation of Data Quality Disparity and Implications for Fair Machine Learning | fairface-citing |
| 11 | 0.784 | 2022 | A survey on bias in visual datasets | fairface-citing |
| 12 | 0.784 | 2023 | Add-Remove-or-Relabel: Practitioner-Friendly Bias Mitigation via Influential Fairness | fairface-citing |
| 13 | 0.783 | 2023 | Enhancing Fairness of Visual Attribute Predictors | fairface-citing |
| 14 | 0.781 | 2024 | FairIF: Boosting Fairness in Deep Learning via Influence Functions with Validation Set Sensitive Attributes | fairface-citing |
| 15 | 0.780 | 2024 | Fairness Testing: A Comprehensive Survey and Analysis of Trends | fairface-citing |

---

## 4. Análise de novidade (sondas Q11–Q14)

Esta seção é o coração do capítulo para a defesa: ela testa se a contribuição da tese já existe na literatura.

### Q11_moo_arch_fairness

> *Query:* Multi-objective architecture search Pareto front accuracy fairness trade-off face attribute classification

- **Similaridade máxima no corpus:** 0.873
- **Veredito automático:** ALTA — possível overlap, **leitura humana obrigatória** dos top-5

Top-5 mais próximos (para leitura manual):

1. (0.873, 2022) FairGRAPE: Fairness-Aware GRAdient Pruning mEthod for Face Attribute Classification
2. (0.848, 2024) FineFACE: Fair Facial Attribute Classification Leveraging Fine-Grained Features
3. (0.842, 2023) F3: Fair and Federated Face Attribute Classification with Heterogeneous Data
4. (0.838, 2023) CAT: Controllable Attribute Translation for Fair Facial Attribute Classification
5. (0.820, 2025) Privacy-preserving face attribute classification via differential privacy

### Q12_pareto_epoch

> *Query:* Pareto-aware epoch selection criterion within multi-objective hyperparameter optimization trial for fairness

- **Similaridade máxima no corpus:** 0.795
- **Veredito automático:** ALTA — possível overlap, **leitura humana obrigatória** dos top-5

Top-5 mais próximos (para leitura manual):

1. (0.795, 2017) Fairness constraints: Mechanisms for fair classification
2. (0.795, 2025) Fairness in Machine Learning: A Review for Statisticians
3. (0.781, 2025) Computational fairness in adaptive neural networks
4. (0.780, 2024) Utility-Fairness Trade-Offs and how to Find Them
5. (0.772, 2017) Algorithmic decision making and the cost of fairness

### Q13_decomposition

> *Query:* Controlled experimental decomposition isolating dataset quality versus model architecture contribution to demographic bias

- **Similaridade máxima no corpus:** 0.870
- **Veredito automático:** ALTA — possível overlap, **leitura humana obrigatória** dos top-5

Top-5 mais próximos (para leitura manual):

1. (0.870, 2024) Dsap: Analyzing Bias Through Demographic Comparison of Datasets
2. (0.841, 2024) Demographic Bias Mitigation at Test-Time Using Uncertainty Estimation and Human-Machine Partnership
3. (0.839, 2024) Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition
4. (0.839, 2023) Metrics for Dataset Demographic Bias: A Case Study on Facial Expression Recognition
5. (0.821, 2024) Demographic bias mitigation at test-time using uncertainty estimation and human–machine partnership

### Q14_recipe_dependent_clean

> *Query:* Recipe-dependent effect of dataset cleaning on margin-based versus softmax-based losses for fairness

- **Similaridade máxima no corpus:** 0.834
- **Veredito automático:** ALTA — possível overlap, **leitura humana obrigatória** dos top-5

Top-5 mais próximos (para leitura manual):

1. (0.834, 2025) Fairness in Machine Learning: A Review for Statisticians
2. (0.813, 2025) Fairness is in the Details : Face Dataset Auditing
3. (0.803, 2017) Fairness constraints: Mechanisms for fair classification
4. (0.797, 2017) Algorithmic decision making and the cost of fairness
5. (0.793, 2024) Utility-Fairness Trade-Offs and how to Find Them

> **Nota metodológica:** similaridade alta NÃO prova overlap (pode ser tema próximo com método distinto); similaridade baixa é evidência mais forte de novidade. A decisão final exige leitura dos PDFs dos top-5 de Q12/Q13 (o delta-núcleo) e cruzamento com sota_review.md §5.0.

---

## 5. Interpretação humana das sondas de novidade (veredito para a tese)

> Esta seção é **leitura humana** dos top-5 (não automática). É o
> material que vai direto para a seção "Posicionamento da Contribuição"
> da dissertação. Data da análise: 2026-05-15.

### Q11 — Busca multi-objetivo de arquitetura para fairness (sim. 0.873)

Top-5 reais: **FairGRAPE** (2022, *gradient pruning* para atributo
facial justo), **FineFACE** (2024, *multi-expert por camada*), **F3**
(2023, *federated*), **CAT** (2023, *attribute translation* generativa),
privacy-DP (2025).

**Veredito:** existe um corpo de trabalho de "arquitetura justa para
classificação de atributo facial" — **mas cada um usa um mecanismo
diferente** (pruning, multi-expert, federated, generativo). **Nenhum é
HPO multi-objetivo de topologia de head com backbone congelado.**
Combinado com o achado de sota_review §5.0 (NeurIPS 2023 "Fair
Architectures", NAS completo em CelebA/VGGFace2), a conclusão é firme:
**"busca de arquitetura justa" NÃO é novidade; o nosso recorte
específico (HPO de head, backbone fixo, FairFace-raça) é o
diferenciador, não o conceito.** → Citar FairGRAPE, FineFACE como
related work; **não reivindicar o conceito como contribuição.**

### Q12 — Critério Pareto-aware best-epoch (sim. 0.795) — DELTA-NÚCLEO

Top-5 reais: "Fairness constraints" (2017), "Fairness in ML: A Review
for Statisticians" (2025), "Computational fairness in adaptive NNs"
(2025), "Utility-Fairness Trade-Offs and how to Find Them" (2024),
"Algorithmic decision making and the cost of fairness" (2017).

**Veredito:** **nenhum dos 5 trata de seleção de epoch dentro de um
trial de HPO multi-objetivo.** São papers de teoria/revisão de
fairness — a similaridade de 0.795 é dirigida por vocabulário
compartilhado (*fairness, trade-off, multi-objective*), não por método.
**Esta é a contribuição com MAIOR probabilidade de novidade em todo o
corpus de 555 documentos.** Ação: ler o PDF de "Utility-Fairness
Trade-Offs and how to Find Them" (2024) — é o único potencialmente
adjacente (frontier-finding), mas pelo título trata da fronteira em si,
não do critério de epoch. **Asset de maior valor para o paper.**

### Q13 — Decomposição controlada dataset vs arquitetura (sim. 0.870)

Top-5 reais: **DSAP** (2024, *comparação demográfica de datasets*),
*Demographic Bias Mitigation at Test-Time* (2024, ×2), *Metrics for
Dataset Demographic Bias: FER case study* (2023/2024).

**Veredito:** há trabalho de **medição de viés de dataset** (DSAP,
métricas FER) e de **mitigação em test-time** — mas **nenhum faz a
isolação experimental controlada** dos fatores (limpeza vs topologia,
mesma seed/split, um fator por vez). Adjacência temática real,
**sem overlap de método**. → Decomposição permanece **defensável como
novidade**, mas a dissertação **deve citar DSAP 2024 e as métricas de
viés de dataset** como related work e explicitar a diferença
(eles medem viés de dataset; nós isolamos a *contribuição causal* de
cada fator para a disparidade).

### Q14 — Efeito recipe-dependent da limpeza (sim. 0.834)

Top-5 reais: revisões gerais de fairness + **"Fairness is in the
Details: Face Dataset Auditing"** (2025).

**Veredito:** o único real adjacente é "Fairness is in the Details"
(2025) — auditoria de dataset facial, próximo do nosso Movimento 1.
**Mas o efeito específico recipe-dependent (CE +0.3pp vs ArcFace
+5.8pp) não está representado.** → Ler+citar "Fairness is in the
Details" (2025); o achado recipe-dependent permanece novidade de
baixo risco.

### Veredito consolidado (cross-check com sota_review §5.0)

| Contribuição candidata | Auditoria semântica | sota_review §5.0 | Veredito final |
|---|---|---|---|
| HPO multi-obj. de arquitetura p/ fairness | Q11=0.873, vários adjacentes | NeurIPS 2023 cobre | ❌ **Não é novidade** — veículo, não contribuição |
| **Critério Pareto-aware best-epoch** | Q12=0.795, **nenhum método igual** | Não documentado | ✅ **Novidade mais forte** — asset principal |
| **Decomposição controlada cleaning×topologia** | Q13=0.870, adjacente sem overlap | Survey 2025 lista como lacuna | ✅ **Defensável** — citar DSAP/FER como related |
| Efeito recipe-dependent da limpeza | Q14=0.834, baixo overlap | Não encontrado | ✅ Novidade de baixo risco |

**Conclusão para a escrita:** o eixo central da tese **deve ser o
critério Pareto-aware best-epoch (Q12) aplicado à decomposição
controlada (Q13)**, com HPO multi-objetivo como *veículo metodológico*
explicitamente posicionado vs FairGRAPE/FineFACE/NeurIPS-2023.

---

## 6. Síntese para a escrita da dissertação

- Esta auditoria + a revisão sistemática (sota_review.md) + a análise
  das referências (fairface_references_analysis.md) formam o
  **capítulo de revisão de literatura** da qualificação.
- A §5 acima é a **seção "Posicionamento da Contribuição"** quase
  pronta — só precisa da leitura dos 3 PDFs sinalizados.
- O ranking por query (§3) alimenta as **subseções temáticas**
  (fairness metrics, contrastivo, losses adaptativas, backbones).
- **Papers de leitura obrigatória antes da submissão** (sinalizados
  pela auditoria): (1) "Utility-Fairness Trade-Offs and how to Find
  Them" 2024 [Q12]; (2) DSAP 2024 [Q13]; (3) "Fairness is in the
  Details: Face Dataset Auditing" 2025 [Q14]; (4) FairGRAPE 2022 e
  FineFACE 2024 [Q11, related work].
- Reprodutibilidade: `python scripts/semantic_search_corpus.py`
  regenera as §1–§4; a §5 é interpretação humana versionada à mão.
