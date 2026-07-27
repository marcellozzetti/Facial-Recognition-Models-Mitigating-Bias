# E-mail ao orientador — Julho/2026

**Para:** Prof. Marcos Gonçalves Quiles
**De:** Marcello Vinícius Alves Ozzetti Cruz
**Assunto:** Qualificação — escrita fechada e cronograma proposto até a defesa

---

Prezado Prof. Marcos,

Escrevo para reportar o fechamento da escrita da monografia de qualificação e submeter o cronograma proposto até a defesa da dissertação. Segue material de apoio anexo (`material_reuniao_orientador_2026-07.pptx`, 6 slides).

## 1. Entrega da monografia de qualificação

A monografia está **consolidada em 5 capítulos**, em conformidade com as normas ABNT NBR 14724 (trabalhos acadêmicos), NBR 10520 (citações autor-data via `biblatex-abnt`) e NBR 6028 (resumo):

- **Cap. 1 — Introdução**: contexto (NIST 2019, FaceScanPaliGemma 75,7 %), problema (gap Latinx de 30 pp), objetivo e contribuições em três eixos.
- **Cap. 2 — Revisão bibliográfica**: 12 seções cobrindo seis frentes teórico-empíricas, dois paradigmas de backbone (convolucional vs. atenção), métricas formais de *fairness* sob o Teorema da Impossibilidade, auditoria de datasets e alternativas de *conditioning*. Corpus consolidado em **104 fichas bibliográficas**, com 53 % das publicações de 2024 ou posteriores, incluindo cinco pré-*prints* de 2026.
- **Cap. 3 — Objetivos, hipóteses e contribuições**: objetivo geral, seis objetivos específicos, seis hipóteses testáveis com critérios formais de confirmação/refutação, e sete contribuições esperadas mapeadas aos três eixos.
- **Cap. 4 — Metodologia**: pipeline em seis etapas, quatro configurações comparativas de *conditioning*, seis *baselines* de mitigação, mecanismo FiLM (Perez et al., 2018) sobre *backbone* ConvNeXt-T (Liu et al., 2022), triangulação de métricas em dois cenários (*race* apenas e *race* × *gender*), aderente à norma **ISO/IEC 19795-10:2024**, e rigor experimental com três sementes independentes.
- **Cap. 5 — Cronograma e riscos**: cronograma mapeado às seis etapas do pipeline e quatro classes de risco com estratégias de mitigação documentadas.

O resumo em português (488 palavras) e o *abstract* em inglês (474 palavras) estão dentro do limite ABNT de 500 palavras, ambos com cinco palavras-chave.

## 2. Highlights técnicos consolidados nesta rodada

- **Rigor bibliográfico**: auditoria completa de 22 afirmações quantitativas contra as fichas primárias, com correção de duas atribuições incorretas identificadas (GRAS e Aguirre).
- **Reforço arquitetural**: nova seção formal discutindo *backbones* em dois paradigmas, com posicionamento correto do ConvNeXt-T como CNN moderna (não paradigma novo).
- **Cobertura de métricas**: seção dedicada ao Teorema da Impossibilidade (Kleinberg et al., 2017) com definições formais canônicas (Dwork et al., 2012; Hardt et al., 2016).
- **Blindagens contra críticas antecipadas**: (i) controle explícito de *pixel information* na Etapa 5, endereçando a tese de Pangelinan (2023); (ii) descritores adicionais de qualidade de imagem (luminância L\*, nitidez, resolução) como salvaguarda contra atribuição puramente sensorial; (iii) *prompt ensembling* na Configuração D como resposta à sensibilidade documentada do CLIP.
- **Visualização Pareto-eficiente** incorporada à triangulação de métricas para diagnóstico visual de dominância entre configurações.
- **Oito figuras** integradas ao Cap. 2, incluindo figura oficial de Liu et al. (2022) reproduzida do repositório *facebookresearch/ConvNeXt* com atribuição direta aos autores.

## 3. Cronograma proposto até a defesa

| Período | Etapa | Marco |
|---|---|---|
| Julho/2026 | — | Solicitação da qualificação ao PPG-CC |
| **Set–Out/2026** | — | **Exame de qualificação** |
| Nov/2026 | Etapa 1 | Infraestrutura + SkinToneNet + preparação dos *datasets* |
| Dez/2026 | Etapa 2 | Matriz pública MST × classes raciais sobre o FairFace |
| Jan–Mar/2027 | Etapa 3 | ConvNeXt-T + FiLM (4 configurações de *ablation*) |
| Abr/2027 | Etapa 4 | Comparação contra 6 *baselines* + triangulação de métricas |
| Mai/2027 | Etapa 5 | Transferência *fair* para *face recognition* (RFW/BFW) |
| Jun/2027 | Etapa 6 | Síntese decompositiva do erro Latinx |
| Nov/2026 – Jul/2027 | — | Redação da dissertação em paralelo |
| **2º sem/2027** | — | **Defesa da dissertação** |

O cronograma preserva folga de contingência de aproximadamente dois meses entre a conclusão da Etapa 6 (junho de 2027) e a janela de defesa, alinhada ao prazo institucional do PPG-CC.

## 4. Próximo passo

Solicito, se possível, uma janela de reunião nesta semana ou início da próxima para revisar o material em conjunto e alinhar a submissão formal da qualificação ao PPG-CC.

Fico à disposição para agendar conforme sua conveniência.

Atenciosamente,

**Marcello Vinícius Alves Ozzetti Cruz**
Mestrado em Ciência da Computação — Unifesp / ICT

---

*Material de apoio anexo: `material_reuniao_orientador_2026-07.pptx` (6 slides, formato 16:9).*
