---
name: dataset-robinson-2020
status_verificacao: VERIFIED
autores: [Joseph P. Robinson, Gennady Livitz, Yann Henon, Can Qin, Yun Fu, Samson Timoner]
ano: 2020
titulo: "Face Recognition: Too Bias, or Not Too Bias?"
venue: "IEEE/CVF CVPR Workshops (CVPRW)"
tipo_publicacao: conference_workshop
arxiv_id: "2002.06483"
doi: null
url_primario: https://arxiv.org/abs/2002.06483
citacoes_google_scholar: null
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-05-25
n_referencias_paper: ~42
lente_disrupcao: cobertura
fonte_leitura: PDF integral extraído via pypdf (pdfs/robinson_2020_bfw.pdf)
---

# BFW: Balanced Faces in the Wild (Robinson et al., 2020)

## 1. Resumo do problema atacado

Sistemas de FR usam **threshold global único** para decisão match/no-
match. **Distribuições de similarity score variam entre subgrupos
demográficos** → threshold global é injusto: favorece a maioria.
Solução proposta: (a) novo dataset **balanceado em 4 etnias × 2
gêneros = 8 subgrupos**, BFW; (b) **threshold adaptativo por
subgrupo**; (c) avaliação humana paralela para comparar viés
humano vs algorítmico.

## 2. Método

### 2.1 Dataset BFW

- **Subgrupos:** 8 = 4 etnias (Asian, Black, Indian, White) × 2
  gêneros (M, F).
- **Composição:** 100 sujeitos por subgrupo × 25 faces/sujeito =
  **800 sujeitos, 20 000 imagens**.
- **Pares:** 921 379 (240 000 positivos + 681 379 negativos).
- **Origem:** sampled de VGGFace2 (≠ RFW que usa MS-Celeb-1M).

### 2.2 Threshold adaptativo

Em vez de threshold global t_g, aprende t_{g_subgrupo} por
sub-população detectada. Mostra melhora simultânea em **fairness +
accuracy** — boost geral.

### 2.3 Avaliação humana

Survey NIH-certified (Protect Humans in Research) demonstra **viés
análogo em percepção humana** — não é só artefato de modelo.

## 3. Datasets e setup experimental

BFW + faces detectadas via SOTA FR systems (encoder CNN).
**Avaliação: face verification 1:1**, não classification.

## 4. Métricas reportadas

FNR (False Negative Rate), FPR (False Positive Rate), EER (Equal
Error Rate), accuracy por subgrupo.

## 5. Resultados principais

- Threshold global produz gaps notáveis entre subgrupos.
- **Subgroup-specific thresholds** reduzem gaps + **boost overall**.
- Viés humano confirmado (não relatado numericamente no excerto).

## 6. Limitações declaradas pelos autores

- **800 sujeitos é pequeno** (RFW tem ~12K identidades; FairFace 108K
  imagens).
- 4 etnias apenas, sem Middle Eastern, Latinx, Southeast Asian.
- 1:1 verification, não classification.

## 7. Limitações que identifiquei

- **Dataset substantialmente menor que RFW** (Neto 2025 destaca:
  "BFW has roughly 4x fewer faces and 15x fewer identities" que RFW).
- **Sampled de VGGFace2** que herda viés do dataset upstream.
- **Threshold adaptativo per-subgroup exige detecção upstream de
  raça** — circular.

## 8. Relação com nossa pesquisa

**Não é diretamente aplicável** (1:1 verification, 4 classes), mas:

1. Reforça evidência convergente: **balanceamento de dataset → não
   resolve fairness** (mesmo padrão que FairFace, RFW, NISTIR 8280).
2. Conceito de **threshold adaptativo per-subgrupo** é
   metodologicamente interessante mas inaplicável diretamente em
   classification softmax (não tem threshold único; tem argmax).
3. Citação para responder Q07 (alternativas ao FairFace): BFW é uma
   das poucas, mas para verification, não classification.

## 9. Pontos para citar

- *"O Balanced Faces in the Wild (BFW), introduzido por Robinson et
  al. (CVPRW 2020), balanceia 4 etnias × 2 gêneros em 8 subgrupos
  (800 sujeitos, 20 000 imagens). Embora ~4× menor em imagens e 15×
  menor em identidades que o RFW (Wang et al., 2019), permanece
  importante por demonstrar empiricamente que thresholds
  subgrupo-específicos melhoram simultaneamente fairness e accuracy
  em face verification."*

## 10. Arquivos relacionados

- PDF: `pdfs/robinson_2020_bfw.pdf` (gitignored).
- Código + dataset: github.com/visionjo/facerec-bias-bfw.
- Entradas relacionadas: [[dataset_wang_2019]] (RFW, alternativa),
  [[neto_2025]] (BFW é benchmark de teste em Neto), [[grother_2019]]
  (NIST FRVT, escala industrial paralela).
- Responde Q07 (datasets não-FairFace) em [[_perguntas]].

## 11. Trabalhos sugeridos pelos autores (Future Work)

Extraído de Conclusion:

- **Expandir BFW para mais etnias** (atualmente 4 = A, B, I, W) —
  Middle Eastern, Latinx, SE Asian estão ausentes. ⚠ Alinhada com
  Q07/Q08 (debate sobre granularidade).
- **Reduzir tamanho do gap entre fairness e performance** via
  **subgroup-specific thresholds** + outras técnicas combinadas.
  ✅ **Alinhada com Q04** (mitigação algorítmica).
- **Estender avaliação humana** para mais demografias e contextos
  (paper testou em pool limitado nos EUA). ❌ Fora do escopo.
- **Auditar viés em soft biometrics** (idade, pose, qualidade).
  ⚠ Parcialmente alinhada.
- **Cross-disciplinary work com ciência social e direito** — viés
  algorítmico tem implicações jurídicas. ❌ Fora do nosso escopo
  experimental.

## 12. Análise crítica do método

### (a) Rigor formal

- **Threshold adaptativo** é proposta operacionalmente clara —
  thresholds subgrupo-específicos via stratified distribution.
- **Métricas FR padrão** (FNR, FPR, EER) bem definidas.
- **Avaliação humana paralela** é metodologicamente sólida (NIH
  certificada).
- **Limitação**: paper workshop CVPRW (5 páginas), análise menos
  profunda que main conference.

### (b) Reprodutibilidade

- ✅ Código + dataset públicos: github.com/visionjo/facerec-bias-bfw.
- ✅ Pares 921 379 (240K positivos + 681K negativos) — protocolo claro.
- ⚠ Dataset pequeno (800 sujeitos) limita generalização estatística.
- ⚠ Hiperparâmetros de threshold adaptativo não detalhados extensivamente.

### (c) Aplicabilidade ao pipeline v3.2

- **FR verification 1:1**, não classification 7-class — não diretamente
  aplicável.
- **Threshold adaptativo per-subgroup** é conceitualmente interessante
  mas inaplicável em softmax classification (argmax, não threshold).
- **BFW útil como dataset alternativo de validação** para H5 (Cap 3) —
  comparar com RFW.

### (d) Design choices justificadas vs assumidas

| Decisão | Justificada? |
|---|---|
| 8 subgrupos (4 race × 2 gender) | ✅ Justificada — intersectional analysis |
| Sample de VGGFace2 | ⚠ Herda viés do upstream |
| 100 sujeitos × 25 faces | ⚠ Choice — pequeno mas controlado |
| Threshold adaptativo per-subgroup | ✅ Justificada — boost simultâneo |
| NIH human evaluation | ✅ Justificada — diferencia bias humano vs algorítmico |
| Workshop venue (CVPRW) | ⚠ Limitação de profundidade vs CVPR main |

### (e) Conexão com R5/R6

- [[buolamwini_2018]]: BFW segue mesma linhagem de auditoria, mas
  para FR não classification.
- [[dataset_wang_2019]] RFW: alternativa em FR com mais identidades
  (12K vs 800). Complementares.
- [[neto_2025]]: usa BFW como benchmark de teste; questiona
  discretização.
- [[hardt_2016]]: thresholds adaptativos per-subgroup violam
  fairness through unawareness mas satisfazem demographic parity
  por construção.
- [[grother_2019]] NIST: escala industrial maior, BFW é versão
  acadêmica multi-subgrupo.
