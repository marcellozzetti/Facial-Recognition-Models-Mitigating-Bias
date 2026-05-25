# Análise das Referências do Paper FairFace (76 refs, 2006–2019)

**Fonte:** Kärkkäinen & Joo, *FairFace* (WACV 2021, arXiv:1908.04913),
seção References. Extraídas integralmente em 2026-05-14.
**Corpus estruturado:** [literature_corpus.csv](literature_corpus.csv)
(76 linhas, colunas: title, authors, year, venue, theme, source).
**Relação com o [sota_review.md](sota_review.md):** este documento cobre
os **ancestrais** do FairFace; o sota_review cobre os **descendentes**
(papers que citam o FairFace).

---

## 1. Distinção metodológica essencial

```
            ┌──────────────────────────────┐
2006–2019   │  76 refs QUE O FAIRFACE CITA │  ← ESTE documento
            │  (ancestrais / fundação)     │     (linhagem intelectual)
            └──────────────────────────────┘
                          │
                    FairFace (2021)
                          │
            ┌──────────────────────────────┐
2021–2026   │  ~900 papers QUE CITAM       │  ← sota_review.md §5.0
            │  FairFace (descendentes)     │     (check de novidade)
            └──────────────────────────────┘
```

**Implicação central para a tese:** a referência mais recente do
FairFace é de **2019**. Toda a fundação intelectual do FairFace
**precede** os métodos que a nossa tese usa:

| Método da nossa tese | Ano de origem | Estava na fundação FairFace? |
|---|---|---|
| ArcFace (margin loss) | 2019 | Não (limítrofe, não citado) |
| Optuna (framework HPO) | 2019 | Não |
| HPO multi-objetivo para fairness | 2022+ | Não |
| AdaFace / MagFace | 2022 | Não |
| Contrastivo supervisionado (SupCon) | 2020 | Não |
| Vision Transformer (ViT) | 2020 | Não |
| Critério Pareto-aware best-epoch | (nosso) | Não |

→ **A contribuição da tese opera num espaço metodológico que não
existia quando o FairFace foi escrito.** Nenhuma das 76 referências
pode pré-emptar o nosso delta. O risco de novidade vive **somente** na
literatura descendente pós-2021 (já parcialmente verificada em
sota_review §5.0: NeurIPS 2023, FineFACE 2024).

---

## 2. Distribuição temática das 76 referências

| Tema | Qtd | Refs representativas |
|---|---:|---|
| Face datasets | 11 | CelebA, VGGFace2, MS-Celeb-1M, MegaFace, LFW, CASIA |
| Generative / augmentation | 8 | StyleGAN, CVAE-GAN, Fader Networks, Attribute2Image |
| Face attribute (não-fairness) | 8 | Han 2018, Kumar 2011, IMDB-WIKI |
| Fairness theory | 8 | Hardt (Equality of Opportunity), Zemel (Fair Representations), Zafar (Fairness Constraints) |
| Face application (ciência social) | 7 | Joo et al. (política, protestos) — agenda do grupo do autor |
| Fairness audit | 6 | **Buolamwini & Gebru (Gender Shades)**, Raji & Buolamwini, Stock & Cisse |
| Face recognition models | 6 | FaceNet, DeepFace, VGGFace, DeepID |
| Face detection | 4 | WIDER Face, Tiny Faces, CNN cascade |
| Fairness debiasing | 3 | Adversarial (Zhang 2018), Alvi 2018, Hendricks 2018 |
| Fairness dataset | 3 | Diversity in Faces, Torralba (Dataset Bias), Shankar (geodiversity) |
| Face alignment | 2 | SDM, LBF |
| Domain theory | 2 | Encyclopedia of Race; Fitzpatrick skin type |
| Fairness attribute-mitigation | 2 | **InclusiveFaceNet**, Das 2018 |
| Backbone / optim / methods / cleaning | 4 | ResNet (He 2016), Adam, t-SNE, Ng&Winkler (cleaning) |

---

## 3. Como o FairFace-era tratava mitigação de viés

As únicas referências de **mitigação** de viés na fundação são:

1. **Adversarial debiasing** — Zhang et al. 2018 [72], Alvi et al. 2018 [1]
2. **Fair representations** — Zemel et al. 2013 [71]
3. **Fairness constraints** — Zafar et al. 2017 [70]
4. **Attribute-level mitigation** — InclusiveFaceNet [47], Das et al. 2018 [12]

**Nenhuma** é baseada em:
- otimização de hiperparâmetros / busca de arquitetura
- otimização multi-objetivo / frente de Pareto
- limpeza geométrica de dataset (multi-face)

→ A nossa família de abordagem (topologia de classificador via HPO +
decomposição controlada) é **disjunta** da família de mitigação da
era FairFace. Posicionamento de related-work limpo e defensável.

---

## 4. Pontos de ancoragem para a dissertação (related work)

Referências canônicas que **devem** ser citadas na dissertação por
serem fundacionais (não por overlap, mas por linhagem):

- **Buolamwini & Gebru 2018 (Gender Shades)** [7] — motivação seminal
  do problema (34.7 pp de gap demográfico em sistemas comerciais).
- **Hardt et al. 2016 (Equality of Opportunity)** [17] — definição
  formal de fairness que fundamenta métricas como Equalized Odds.
- **Zemel et al. 2013 (Learning Fair Representations)** [71] — origem
  da linha de "fair representation learning".
- **He et al. 2016 (ResNet)** [18] — o backbone que usamos.
- **Ng & Winkler 2014 (cleaning large face datasets)** [40] — único
  precedente de *dataset cleaning* na fundação; **mas trata ruído de
  identidade, não ambiguidade geométrica multi-face** — diferenciação
  explícita do nosso Movimento 1.
- **Verma & Rubin 2018 (Fairness definitions explained)** [61] —
  taxonomia de definições de fairness para o capítulo de fundamentação.

---

## 5. Próximo passo (passada definitiva — Diretriz 8)

O `literature_corpus.csv` agora tem **76 linhas seed** (ancestrais).
Para a passada definitiva de novidade, falta o conjunto **descendente**
(papers que citam o FairFace, ~900). Como obter:

1. **Semantic Scholar / Google Scholar "Cited by"** do FairFace
   (paper id arXiv:1908.04913) → exportar BibTeX/CSV.
2. Concatenar ao `literature_corpus.csv` (mesmo schema, `source=fairface-citing`).
3. Rodar `scripts/semantic_search_corpus.py --input docs/literature_corpus.csv`
   — gera embeddings, indexa com FAISS, roda as 10 queries semânticas
   (incluindo a Q de overlap do nosso delta), escreve
   `docs/literature_semantic_audit.md`.

**Bloqueio atual:** o export "cited by" (~900) precisa ser fornecido —
não é extraível por web fetch (requer conta Semantic Scholar/Scholar ou
a API com chave). As 76 ancestrais já estão estruturadas e prontas.

---

## 6. Conclusão

- As 76 referências do FairFace são **2006–2019** → não podem conter
  nossa contribuição (métodos pós-2020). **Isto fortalece o argumento
  de novidade**, não o enfraquece.
- A família de mitigação da era FairFace (adversarial, fair-rep,
  constraints) é **disjunta** da nossa (HPO de topologia + decomposição).
- O risco de novidade real está **exclusivamente** nos descendentes
  pós-2021, já parcialmente coberto em sota_review §5.0.
- `literature_corpus.csv` está pronto como seed; falta o export
  "cited-by" para a passada definitiva (Diretriz 8).
