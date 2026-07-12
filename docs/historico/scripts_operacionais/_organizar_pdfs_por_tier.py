"""Organiza PDFs em subpastas por tier de relevancia para NotebookLM.

Cria 4 pastas em docs/ativo/04_pesquisa_bibliografica/pdfs_por_tier/:
- tier_1_critico/   (~14 PDFs) - leitura integral obrigatoria
- tier_2_alto/      (~33 PDFs) - apoio direto tese
- tier_3_medio/     (~30 PDFs) - cobertura ampla
- tier_4_baixo/     (~24 PDFs) - contextual

NotebookLM limites:
- Free: 50 fontes por notebook -> usar tier 1 + tier 2 (47 PDFs)
- Pro: 300 fontes -> pode usar todos

Uso:
    python _organizar_pdfs_por_tier.py
    -> cria pastas com copias dos PDFs (nao mexe nos originais)
"""

from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "04_pesquisa_bibliografica"
PDFS_DIR = CORPUS_DIR / "pdfs"
OUT_BASE = CORPUS_DIR / "pdfs_por_tier"


# TIER 1 - Critico (~14 fichas)
# Camada 1 do pente fino: 12 forte favoravel + 2 conflito forte
TIER_1 = [
    # Forte favoravel (12)
    "pereira_2026",                  # SkinToneNet - Etapa 1
    "dataset_karkkainen_2021",       # FairFace - dataset central
    "aldahoul_2024",                 # FaceScanPaliGemma - SOTA atual
    "schumann_2023",                 # MST canonico
    "perez_2018",                    # FiLM - mecanismo central
    "hardt_2016",                    # EO/EOD canonico
    "kleinberg_2017",                # Impossibility theorem
    "madras_2018",                   # LAFTR transferencia
    "sagawa_2020",                   # Group DRO baseline
    "park_2022",                     # FSCL+ baseline
    "lin_2022",                      # FairGRAPE + validacao baseline
    "luo_2024_fairclip",             # FairCLIP - C7 baseline
    # Conflito forte (2)
    "pangelinan_2023",               # Refutacao central - H6
    "neto_2025",                     # Continuous labels - limitacao
]

# TIER 2 - Alto (~33 fichas)
# Camada 2 favorable + papers fundadores criticos
TIER_2 = [
    # Fundadores teoricos
    "buolamwini_2018",               # Gender Shades - marco fundador
    "fuentes_2019",                  # AAPA Statement - race construto
    "lewontin_1972",                 # Apportionment 85/6/8
    "grother_2019",                  # NIST FRVT
    "zemel_2013",                    # LFR Test-of-Time
    "zhang_2018",                    # Adversarial baseline
    "aguirre_2023",                  # Multi-task fair empirico
    # Datasets
    "dataset_wang_2019",             # RFW
    "dataset_robinson_2020",         # BFW
    "dataset_bupt_2019",             # BUPT/MBN - precedente skin tone
    "dataset_hazirbas_2021",         # CCv1
    "porgali_2023_ccv2",             # CCv2 Meta
    "fitzpatrick_1988",              # FST historico
    # Baselines de mitigacao (criticos para Cap 2)
    "manzoor_2024",                  # FineFACE
    "bhaskaruni_2019",               # Ensemble
    "dehdashtian_2024",              # U-FaTE trade-off
    "liu_2025",                      # BNMR FAccT
    # FR fundadores
    "schroff_2015_facenet",          # FaceNet
    "wang_2018_cosface",             # CosFace
    "deng_2019_arcface",             # ArcFace
    "meng_2021_magface",             # MagFace
    "kim_2022_adaface",              # AdaFace
    # Conflitos moderados + refutacoes
    "dooley_2022",                   # NAS arquitetural
    "kolla_2022",                    # 16 experimentos
    "rethinking_assumptions_2021",   # Refutacao balanceamento
    "image_distortions_2021",        # Confounder
    "occlusion_bias_2024",           # Confounder
    # Surveys top
    "survey_kotwal_2025",            # Survey FR fairness
    "survey_mehrabi_2021",           # Survey ML fairness canonico
    "survey_racial_bias_fr_2024",    # Survey Durham
    "survey_fairness_vision_lang_2024",  # Survey PUCRS Brasil
    # CLIP/VLM principais
    "dehdashtian_2024_fairerclip",   # FairerCLIP ICLR
    "bendvlm_2024",                  # BendVLM test-time
    "lafargue_2025",                 # EU AI Act
]

# TIER 3 - Medio (~30 fichas)
# Caminhos alternativos + auditoria adicional
TIER_3 = [
    # Auditoria avancada
    "dominguez_2024",                # DSAP
    "reliable_demo_inference_2025",  # DAI pipeline
    # VLM/CLIP Track I (cobertura)
    "joint_vl_2024",
    "unified_debiasing_vlm_2024",
    "closed_form_debias_2026",
    "bias_subspace_2025",
    "fair_residuals_vlm_2025",
    "biopro_2025",
    "lin_2025_aiface",
    "gras_2025",
    "indicfairface_2026",
    "mllm_face_verification_2026",
    "evaluating_lvlm_2024",
    "benchmark_lvlm_2026",
    # Conditioning moderno Track J
    "tian_2024_fairvit",
    "bian_2025_lorafair",
    "fairlora_2024",
    "fairness_lora_2024",
    "zhao_2025_aimfair",
    # Mitigacao algoritmica adicional
    "ramachandran_2024",
    "provable_adversarial_ssl_2024",
    "raumanns_2024",
    "enhancing_visual_attributes_2022",
    # Conflito moderado adicional
    "robustness_face_detection_2022",
    # Surveys CV/multimodal
    "survey_cv_fairness_2024",
    "survey_multimodal_fairness_2024",
    "survey_llm_bias_2024",
    "survey_face_recognition_2022",
    # FR fundadores adicionais
    "range_loss_2016",
    # Causal/agnostic
    "counterfactual_fairness_iclr2025",
    "demographic_agnostic_2025",
    # Skin tone classifier specifics
    "mst_kd_2024",
    "fairface_challenge_eccv2020",
]

# TIER 4 - Baixo (~24 fichas)
# Track L auxiliar + neutras contextuais
TIER_4 = [
    # Track L synthetic
    "synthetic_face_2024",
    "variface_2024",
    "frcsyn_2024",
    "massively_annotated_2024",
    "fairimagen_neurips2025",
    "fairer_datasets_2024",
    # Track L federated/privacy
    "dp_fedface_2024",
    "voidface_2025",
    "federated_fairness_survey_2025",
    # Track L cross-domain
    "fairdomain_2024",
    "face4fairshifts_2025",
    # Track L explainability
    "explainable_fr_2024",
    # Track L post-hoc/calibration
    "post_comparison_2020",
    "faircal_2021",
    "score_normalization_2024",
    "fair_sight_2025",
    # Histories sociais
    "massey_martin_2003",
    # Outras contextuais
    "racial_bias_dataset_2017",
    "survey_long_tail_2022",
]

TIERS = {
    "tier_1_critico": TIER_1,
    "tier_2_alto": TIER_2,
    "tier_3_medio": TIER_3,
    "tier_4_baixo": TIER_4,
}


def main() -> None:
    if OUT_BASE.exists():
        print(f"Limpando pasta antiga {OUT_BASE}")
        shutil.rmtree(OUT_BASE)

    ok, faltam = [], []
    total = 0
    for tier_name, fichas in TIERS.items():
        tier_dir = OUT_BASE / tier_name
        tier_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{tier_name}] - {len(fichas)} PDFs:")
        for stem in fichas:
            src = PDFS_DIR / f"{stem}.pdf"
            if not src.exists():
                # heuristica: arquivo com nome diferente
                first_segment = stem.split("_")[0]
                cands = list(PDFS_DIR.glob(f"{first_segment}*.pdf"))
                if cands:
                    src = cands[0]
            if not src.exists():
                faltam.append((tier_name, stem))
                print(f"  [FALTA] {stem}.pdf nao encontrado")
                continue
            dst = tier_dir / f"{stem}.pdf"
            shutil.copy2(src, dst)
            ok.append(stem)
            total += 1

    print(f"\n{'='*50}")
    print(f"Total copiados: {total}")
    print(f"Faltando: {len(faltam)}")
    if faltam:
        for tier, stem in faltam:
            print(f"  [{tier}] {stem}")
    print(f"\nPasta: {OUT_BASE}")
    print(f"\nPara NotebookLM:")
    print(f"  Free (50 fontes): use tier_1 + tier_2 ({len(TIER_1) + len(TIER_2)} PDFs)")
    print(f"  Pro (300 fontes): use todos os tiers ({total} PDFs)")


if __name__ == "__main__":
    main()
