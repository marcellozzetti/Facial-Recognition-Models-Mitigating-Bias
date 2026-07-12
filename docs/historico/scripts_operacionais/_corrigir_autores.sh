#!/bin/bash
# Corrige campo 'autores' em fichas C-level com flag 'a verificar'.
# Autorias verificadas via leitura do PDF (primeira pagina).

set -euo pipefail
D="docs/ativo/04_pesquisa_bibliografica"

# array com pares "ficha|autores"
declare -a CORR=(
  "benchmark_lvlm_2026|Xuwei Tan (Ohio State University), Ziyu Hu (Stevens Institute of Technology), Xueru Zhang (Ohio State University)"
  "bendvlm_2024|Walter Gerych (MIT), Haoran Zhang (MIT), Kimia Hamidieh (MIT), Eileen Pan (MIT), Maanas Sharma (MIT), Thomas Hartvigsen (University of Virginia), Marzyeh Ghassemi (MIT)"
  "bias_subspace_2025|Dachuan Zhao (Harvard), Weiyue Li (Harvard), Zhenda Shen (Harvard), Yushu Qiu (Harvard), Bowen Xu (Harvard), Haoyu Chen (Harvard), Yongchao Chen (MIT + Harvard)"
  "biopro_2025|Yujie Lin (Xiamen University), Jiayao Ma (Xiamen), Qingguo Hu (Xiamen), Derek F. Wong (University of Macau), Jinsong Su (Xiamen)"
  "closed_form_debias_2026|Tangzheng Lian (King's College London), Guanyu Hu (Queen Mary University of London), Yijing Ren (King's College London), Dimitrios Kollias (Queen Mary), Oya Celiktutan (King's College London)"
  "counterfactual_fairness_iclr2025|Bowei Tian (University of Maryland), Ziyao Wang (UMD), Shwai He (UMD), Wanghao Ye (UMD), Guoheng Sun (UMD), Yucong Dai (Clemson University), Yongkai Wu (Clemson), Ang Li (UMD)"
  "demographic_agnostic_2025|Zhongteng Cai (Ohio State University), Mohammad Mahdi Khalili (OSU), Xueru Zhang (OSU)"
  "evaluating_lvlm_2024|Xuyang Wu (Santa Clara University), Yuan Wang (Santa Clara), Hsin-Tai Wu (DOCOMO Innovations), Zhiqiang Tao (Rochester Institute of Technology), Yi Fang (Santa Clara University)"
  "explainable_fr_2024|Yuhang Lu (EPFL Switzerland), Zewei Xu (EPFL), Touradj Ebrahimi (EPFL)"
  "fair_residuals_vlm_2025|Jian Lan (LMU Munich + MCML, corresponding), Udo Schlegel (LMU + MCML), Gengyuan Zhang (LMU + MCML), Tanveer Hannan (LMU + MCML), Haokun Chen (LMU), Thomas Seidl (LMU + MCML)"
  "fair_sight_2025|Arya Fayyazi (University of Southern California), Mehdi Kamal (USC), Massoud Pedram (USC)"
  "fairlora_2024|Rohan Sukumaran (Mila + Université de Montréal), Aarash Feizi (Mila + McGill), Adriana Romero-Soriano (Mila + McGill), Golnoosh Farnadi (Mila + UdeM)"
  "fairness_lora_2024|Zhoujie Ding (Stanford), Ken Ziyu Liu (Stanford), Pura Peetathawatchai (Stanford), Berivan Isik (Stanford), Sanmi Koyejo (Stanford)"
  "frcsyn_2024|Ivan DeAndres-Tame (UAM Madrid), Ruben Tolosana (UAM), Pietro Melzi (UAM), Ruben Vera-Rodriguez (UAM), Minchul Kim (MSU), Christian Rathgeb (Hochschule Darmstadt), Xiaoming Liu (MSU), Aythami Morales (UAM), Julian Fierrez (UAM), Javier Ortega-Garcia (UAM), et al."
  "gras_2025|Shaivi Malik (Guru Gobind Singh Indraprastha University + AI Institute USC), Hasnat Md Abdullah (AI Institute USC + University of Illinois Urbana-Champaign), Sriparna Saha (AI Institute USC + IIT Patna, corresponding), Amit Sheth (AI Institute University of South Carolina)"
  "indicfairface_2026|Aarish Shah Mohsin (Aligarh Muslim University), Mohammed Tayyab Ilyas Khan (Aligarh Muslim University), Mohammad Nadeem (Aligarh Muslim University), Shahab Saquib Sohail (Jamia Hamdard), Erik Cambria (Nanyang Technological University Singapore), Jiechao Gao"
  "joint_vl_2024|Haoyu Zhang (National University of Singapore), Yangyang Guo (NUS), Mohan Kankanhalli (NUS)"
  "mllm_face_verification_2026|Ünsal Öztürk (Idiap Research Institute Switzerland), Hatef Otroshi Shahreza (Idiap), Sébastien Marcel (Idiap)"
  "mst_kd_2024|Eduarda Caldeira (INESC TEC + University of Porto), Jaime S. Cardoso (INESC TEC + Univ. Porto), Ana F. Sequeira (INESC TEC + Univ. Porto), Pedro C. Neto (INESC TEC + Univ. Porto)"
  "reliable_demo_inference_2025|Alexandre Fournier-Montgieux (Université Paris-Saclay, CEA, LIST), Hervé Le Borgne (Paris-Saclay CEA LIST), Adrian Popescu (Paris-Saclay CEA LIST), Bertrand Luvison (Paris-Saclay CEA LIST)"
  "robustness_face_detection_2022|Samuel Dooley (University of Maryland), George Z. Wei (University of Massachusetts Amherst), Tom Goldstein (UMD), John P. Dickerson (UMD)"
  "score_normalization_2024|Yu Linghu (University of Zurich), Tiago de Freitas Pereira (ams OSRAM), Christophe Ecabert (Idiap Research Institute), Sébastien Marcel (Idiap), Manuel Günther (University of Zurich)"
  "unified_debiasing_vlm_2024|Hoin Jung (Purdue University), Taeuk Jang (Purdue), Xiaoqian Wang (Purdue)"
  "occlusion_bias_2024|Rafael M. Mamede (University of Porto + INESC TEC), Pedro C. Neto (Univ. Porto + INESC TEC), Ana F. Sequeira (Univ. Porto + INESC TEC)"
  "survey_cv_fairness_2024|Sepehr Dehdashtian, Ruozhen He, Yi Li, Guha Balakrishnan, Nuno Vasconcelos (Fellow IEEE), Vicente Ordonez (Member IEEE), Vishnu Naresh Boddeti (Member IEEE)"
  "survey_multimodal_fairness_2024|Tosin Adewumi (LTU Sweden), Lama Alkhaled (LTU), Namrata Gurung (QualityMinds GmbH Germany), Goya van Boven (Utrecht University Netherlands), Irene Pagliai (University of Göttingen)"
  "fairer_datasets_2024|Alexandre Fournier-Montgieux (Université Paris-Saclay, CEA, LIST), Michael Soumm (Paris-Saclay CEA LIST), Adrian Popescu (Paris-Saclay CEA LIST), Bertrand Luvison (Paris-Saclay CEA LIST), Hervé Le Borgne (Paris-Saclay CEA LIST)"
)

count=0
for entry in "${CORR[@]}"; do
    ficha="${entry%%|*}"
    autores="${entry#*|}"
    file="$D/${ficha}.md"
    if [ ! -f "$file" ]; then
        echo "FALTA $file"; continue
    fi
    # escapa caracteres especiais para sed
    autores_esc=$(printf '%s' "$autores" | sed -e 's/[\/&]/\\&/g')
    sed -i -E "s|^autores: \[a verificar( nome principal)?\]|autores: [${autores_esc}]|" "$file"
    count=$((count+1))
done

echo "Corrigidas: $count"
