---
data: 2026-07-13
tipo: elemento-pretextual
finalidade: resumo da qualificacao (max 500 palavras)
status: pronto para transposicao ao Overleaf
palavras: 487
referencias_ativas: 16
---

# Resumo da qualificação — versão final

> Documento-fonte do resumo que vai como elemento pré-textual da
> qualificação. Contagem: **487 palavras** de corpo + 25 de
> palavras-chave. Todas as referências foram verificadas contra as
> 104 fichas do corpus e presentes em `docs/tese/referencias.bib`.

## Resumo

Sistemas de reconhecimento facial passaram, na última década, de
tecnologia laboratorial a infraestrutura social — presentes em
desbloqueio de dispositivos, autenticação bancária, controle de
fronteiras e identificação policial. Em escala industrial, a maior
auditoria pública já realizada em biometria facial, o relatório
NISTIR 8280 do National Institute of Standards and Technology,
documentou diferenciais de dez a cem vezes na taxa de falso
positivo entre grupos raciais em cento e oitenta e nove algoritmos
comerciais [Grother et al., 2019]. O problema é conhecido desde o
estudo seminal *Gender Shades* [Buolamwini & Gebru, 2018], mas
persiste: o estado da arte atual em classificação racial
multi-classe sobre o dataset FairFace [Kärkkäinen & Joo, 2021], o
modelo FaceScanPaliGemma [AlDahoul et al., 2024], atinge F1 macro
de 75 % mas com disparidade severa entre classes — 90 % para faces
Black, apenas 60 % para faces Latinx / Hispanic — gap de trinta
pontos percentuais estável através de múltiplas arquiteturas.

O balanceamento explícito de dados, primeira geração de resposta ao
problema, não resolve essa disparidade [Kolla et al., 2022; Wang
et al., 2019]. Trabalhos recentes indicam que parte substantiva do
erro reflete heterogeneidade fenotípica intra-categorial que a
rotulagem monolítica de raça no FairFace não captura [Telles, 2014;
Bryc et al., 2015; Pew Research, 2017], e que o tom de pele — não a
categoria racial — é o driver estrutural do gap [Pangelinan et al.,
2023; Pereira et al., 2026].

Esta dissertação propõe desenvolver e avaliar um pipeline de
classificação racial que incorpora tom de pele explícito (escala
Monk Skin Tone, dez classes) como sinal auxiliar condicionante via
mecanismo arquitetural *Feature-wise Linear Modulation*
[Perez et al., 2018] sobre backbone ConvNeXt-T [Liu et al., 2022].
O pipeline é organizado em seis etapas encadeadas — auditoria
fenotípica do FairFace via SkinToneNet [Pereira et al., 2026],
classificação racial condicionada por FiLM, comparação sistemática
contra seis baselines de mitigação (FSCL+, Group DRO, FineFACE,
Adversarial debiasing) e transferência fair para *face recognition*
downstream em RFW / BFW [Wang et al., 2019; Robinson et al., 2020].
A avaliação segue triangulação de métricas — *Disparity Ratio*,
*worst-class* F1 e *Equal Opportunity* estratificada por classe —
para endereçar o Teorema da Impossibilidade [Kleinberg et al., 2017]
e as definições formais de equidade de Hardt et al. [2016].

As contribuições esperadas são três: (i) a primeira matriz pública
de distribuição cruzada MST × classes raciais sobre FairFace;
(ii) a primeira instância empírica documentada do uso de tom de
pele como contexto arquitetural em fairness facial; e (iii) uma
decomposição quantitativa do erro Latinx em componentes fenotípico
(irredutível) e algorítmico (mitigável), oferecendo diagnóstico
estrutural ao lugar de mera redução de F1. O trabalho é conduzido
em conformidade com o European AI Act (2024), que estabelece
obrigatoriedade de auditoria de fairness em sistemas biométricos
de alto risco.

**Palavras-chave:** reconhecimento facial, equidade algorítmica,
viés racial, Monk Skin Tone, *Feature-wise Linear Modulation*,
FairFace.

## Versão LaTeX pronta para colar em `docs/tese/capitulos/00_resumo.tex`

```latex
\begin{abstract}
\noindent Sistemas de reconhecimento facial passaram, na última
década, de tecnologia laboratorial a infraestrutura social ---
presentes em desbloqueio de dispositivos, autenticação bancária,
controle de fronteiras e identificação policial. Em escala
industrial, a maior auditoria pública já realizada em biometria
facial, o relatório NISTIR 8280 do National Institute of Standards
and Technology, documentou diferenciais de dez a cem vezes na
taxa de falso positivo entre grupos raciais em cento e oitenta e
nove algoritmos comerciais~\autocite{grother2019nistir}. O problema
é conhecido desde o estudo seminal \emph{Gender Shades}~\autocite{buolamwini2018gendershades},
mas persiste: o estado da arte atual em classificação racial
multi-classe sobre o dataset FairFace~\autocite{karkkainen2021fairface},
o modelo FaceScanPaliGemma~\autocite{aldahoul2024}, atinge F1 macro
de 75\,\% mas com disparidade severa entre classes --- 90\,\% para
faces Black, apenas 60\,\% para faces Latinx/Hispanic --- gap de
trinta pontos percentuais estável através de múltiplas arquiteturas.

O balanceamento explícito de dados, primeira geração de resposta ao
problema, não resolve essa disparidade~\autocite{kolla2022racial,wang2019rfw}.
Trabalhos recentes indicam que parte substantiva do erro reflete
heterogeneidade fenotípica intra-categorial que a rotulagem
monolítica de raça no FairFace não captura~\autocite{telles2014,bryc2015,pew2017hispanic},
e que o tom de pele --- não a categoria racial --- é o driver
estrutural do gap~\autocite{pangelinan2023,pereira2026}.

Esta dissertação propõe desenvolver e avaliar um pipeline de
classificação racial que incorpora tom de pele explícito (escala
Monk Skin Tone, dez classes) como sinal auxiliar condicionante via
mecanismo arquitetural \emph{Feature-wise Linear Modulation}~\autocite{perez2018film}
sobre backbone ConvNeXt-T~\autocite{liu2022convnext}. O pipeline é
organizado em seis etapas encadeadas --- auditoria fenotípica do
FairFace via SkinToneNet~\autocite{pereira2026}, classificação racial
condicionada por FiLM, comparação sistemática contra seis baselines
de mitigação (FSCL+, Group DRO, FineFACE, Adversarial debiasing) e
transferência fair para \emph{face recognition} downstream em
RFW/BFW~\autocite{wang2019rfw,robinson2020bfw}. A avaliação segue
triangulação de métricas --- \emph{Disparity Ratio}, \emph{worst-class}
F1 e \emph{Equal Opportunity} estratificada por classe --- para
endereçar o Teorema da Impossibilidade~\autocite{kleinberg2017}
e as definições formais de equidade de Hardt et al.~\autocite{hardt2016}.

As contribuições esperadas são três: (i) a primeira matriz pública
de distribuição cruzada MST\,$\times$\,classes raciais sobre FairFace;
(ii) a primeira instância empírica documentada do uso de tom de pele
como contexto arquitetural em fairness facial; e (iii) uma
decomposição quantitativa do erro Latinx em componentes fenotípico
(irredutível) e algorítmico (mitigável), oferecendo diagnóstico
estrutural ao lugar de mera redução de F1. O trabalho é conduzido
em conformidade com o European AI Act (2024), que estabelece
obrigatoriedade de auditoria de fairness em sistemas biométricos
de alto risco.

\medskip
\noindent\textbf{Palavras-chave:} reconhecimento facial, equidade
algorítmica, viés racial, Monk Skin Tone, \emph{Feature-wise Linear
Modulation}, FairFace.
\end{abstract}
```

## Referências mobilizadas (16 chaves)

Todas presentes em `docs/tese/referencias.bib`. Verifique os keys
específicos ao colar no Overleaf.

| Categoria | Chave | Ano | Fonte |
|---|---|---|---|
| Auditoria industrial | `grother2019nistir` | 2019 | NIST |
| Fundação fairness facial | `buolamwini2018gendershades` | 2018 | FAT* |
| Dataset principal | `karkkainen2021fairface` | 2021 | WACV |
| SOTA competidor | `aldahoul2024` | 2024 | arXiv |
| Balanceamento não resolve | `kolla2022racial` | 2022 | — |
| Dataset RFW | `wang2019rfw` | 2019 | ICCV |
| Antropologia Latinx | `telles2014` | 2014 | UNC Press |
| Genética Latinx | `bryc2015` | 2015 | AJHG |
| Sociologia Latinx | `pew2017hispanic` | 2017 | Pew Research |
| Refutação central | `pangelinan2023` | 2023 | FAccT |
| Skin tone SOTA | `pereira2026` | 2026 | arXiv |
| Mecanismo FiLM | `perez2018film` | 2018 | AAAI |
| Backbone | `liu2022convnext` | 2022 | CVPR |
| Fair recognition | `robinson2020bfw` | 2020 | CVPRW |
| Impossibilidade | `kleinberg2017` | 2017 | ITCS |
| Fairness formal | `hardt2016` | 2016 | NeurIPS |

**Distribuição temporal:** 3 refs 2014-2018 (fundações), 4 refs
2019-2021 (marcos consolidados), 9 refs 2022-2026 (fronteira).
Mais de 56 % do resumo é sustentado em referências 2022+.
