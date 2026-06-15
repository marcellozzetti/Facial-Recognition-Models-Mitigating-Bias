---
name: dp-fedface-2024
status_verificacao: VERIFIED
autores: [Wenjing Wang, Si Li (Beijing University of Posts and Telecommunications)]
ano: 2024
titulo: "DP-FedFace: Privacy-Preserving Facial Recognition in Real Federated Scenarios"
venue: "ACM CIKM 2024 — 33rd ACM International Conference on Information and Knowledge Management, Boise, ID, USA, Oct 21-25 2024"
tipo_publicacao: conference
arxiv_id: null
doi: "10.1145/3627673.3679901"
url_primario: https://dl.acm.org/doi/10.1145/3627673.3679901
citacoes_semantic_scholar: null
data_verificacao_citacoes: 2026-06-15
n_paginas: 5
lente_disrupcao: cobertura
fonte_leitura: PDF baixado manualmente pelo Marcello via VPN institucional (pdfs/dp_fedface_2024.pdf). Validação Nível 2 (Camada 2) em 2026-06-15 — abstract, introdução e contribuições lidos via pdftotext.
---

# DP-FedFace — Privacy-Preserving Federated FR (Wang & Li, CIKM 2024)

> **Differential Privacy + Federated Learning** para FR em cenário
> realista: **um identity por cliente** (cada dispositivo só tem
> faces do próprio dono). Usa **DCT block** para transformar imagens
> ao domínio de frequência e remove componentes low-frequency
> visualmente críticos mas menos importantes para reconhecimento.
> Mecanismo **learnable de privacy cost allocation**.

## 1-2. Resumo + método

- **Problema**: FedFace original requer que clientes transmitam
  identity proxies ao servidor — viola o protocolo de federated
  learning.
- **Cenário realista**: cada cliente possui imagens de **uma única
  identidade** (a sua) — comum em deploy real.
- **Contribuição**: framework estendendo FedFace que converte dados
  privados em dados privacy-preserving:
  1. **Block Discrete Cosine Transform (DCT)** — shift para
     frequency domain
  2. **Remoção estratégica** de low-frequency components (ricos em
     visual info mas menos críticos para FR)
  3. **Learnable privacy cost allocation** — adiciona ruído aos
     features de frequência, balanceando privacy × utility
  4. Clientes retêm **apenas features mascarados** (frequency
     domain), não imagens originais
- **Attack simulations** confirmam robustez da proteção.

## 3-5. Datasets, métricas, resultados

- **Métricas**: trade-off privacy vs FR accuracy.
- **Achado**: balanço favorável privacy ↔ performance.

## 7. Aplicação ao pipeline v3.2

- **Direção complementar** (não testada nesta dissertação) — nossa
  tese é centralized, não federated.
- **Cap 1** (Introdução): contexto regulatório (LGPD, GDPR, EU AI
  Act) sobre tratamento de dados biométricos.
- **Cap 4** (Discussão): direção futura — privacy-preserving FR
  com diff. privacy.

## 8. Citar

- *"DP-FedFace (Wang & Li, CIKM 2024) endereça o cenário realista
  de federated face recognition em que cada cliente possui imagens
  de uma única identidade — comum em dispositivos pessoais — via
  combinação de differential privacy com transformada DCT block e
  mecanismo learnable de privacy cost allocation. O resultado
  demonstra viabilidade de FR distribuído em conformidade com
  requisitos regulatórios crescentes (LGPD, GDPR, EU AI Act) sobre
  dados biométricos."*

## 9-12.

PDF: `pdfs/dp_fedface_2024.pdf`. Conexões:
[[lafargue_2025]] (EU AI Act regulatório), [[voidface_2025]] (privacy
FR paralelo), [[federated_fairness_survey_2025]] (survey federated
fairness).
