import argparse
from collections import Counter


# 🔥 ROI → Network (DAN 추가 버전)
ROI_TO_NETWORK = {

    # -------------------------
    # Level 1: Sensory / Somatomotor
    # -------------------------
    "Postcentral_L": "Sensory",
    "Postcentral_R": "Sensory",
    "Precentral_L": "Sensory",
    "Precentral_R": "Sensory",
    "Paracentral_Lobule_L": "Sensory",
    "Paracentral_Lobule_R": "Sensory",

    "Heschl_L": "Sensory",
    "Heschl_R": "Sensory",
    "Temporal_Sup_L": "Sensory",
    "Temporal_Sup_R": "Sensory",

    "Calcarine_L": "Sensory",
    "Calcarine_R": "Sensory",
    "Cuneus_L": "Sensory",
    "Cuneus_R": "Sensory",
    "Lingual_L": "Sensory",
    "Lingual_R": "Sensory",
    "Occipital_Sup_L": "Sensory",
    "Occipital_Sup_R": "Sensory",
    "Occipital_Mid_L": "Sensory",
    "Occipital_Mid_R": "Sensory",
    "Occipital_Inf_L": "Sensory",
    "Occipital_Inf_R": "Sensory",
    "Fusiform_L": "Sensory",
    "Fusiform_R": "Sensory",

    "Rolandic_Oper_L": "Sensory",
    "Rolandic_Oper_R": "Sensory",
    "Supp_Motor_Area_L": "Sensory",
    "Supp_Motor_Area_R": "Sensory",

    # -------------------------
    # Level 2: DAN + Salience (Merged)
    # -------------------------
    "Frontal_Sup_L": "DANnSalience",
    "Frontal_Sup_R": "DANnSalience",
    "Frontal_Mid_L": "DANnSalience",
    "Frontal_Mid_R": "DANnSalience",
    "Parietal_Sup_L": "DANnSalience",
    "Parietal_Sup_R": "DANnSalience",
    "SupraMarginal_L": "DANnSalience",
    "SupraMarginal_R": "DANnSalience",

    "Insula_L": "DANnSalience",
    "Insula_R": "DANnSalience",
    "Cingulum_Ant_L": "DANnSalience",
    "Cingulum_Ant_R": "DANnSalience",

    # -------------------------
    # Level 3: Frontoparietal (Control)
    # -------------------------
    "Frontal_Sup_Orb_L": "Frontoparietal",
    "Frontal_Sup_Orb_R": "Frontoparietal",
    "Frontal_Mid_Orb_L": "Frontoparietal",
    "Frontal_Mid_Orb_R": "Frontoparietal",
    "Frontal_Inf_Oper_L": "Frontoparietal",
    "Frontal_Inf_Oper_R": "Frontoparietal",
    "Frontal_Inf_Tri_L": "Frontoparietal",
    "Frontal_Inf_Tri_R": "Frontoparietal",
    "Frontal_Inf_Orb_L": "Frontoparietal",
    "Frontal_Inf_Orb_R": "Frontoparietal",
    "Frontal_Sup_Medial_L": "Frontoparietal",
    "Frontal_Sup_Medial_R": "Frontoparietal",

    # -------------------------
    # Level 4: DMN
    # -------------------------
    "Frontal_Med_Orb_L": "DMN",
    "Frontal_Med_Orb_R": "DMN",
    "Cingulum_Post_L": "DMN",
    "Cingulum_Post_R": "DMN",
    "Angular_L": "DMN",
    "Angular_R": "DMN",
    "Precuneus_L": "DMN",
    "Precuneus_R": "DMN",
    "Parietal_Inf_L": "DMN",
    "Parietal_Inf_R": "DMN",

    # -------------------------
    # Limbic
    # -------------------------
    "Olfactory_L": "Limbic",
    "Olfactory_R": "Limbic",
    "Hippocampus_L": "Limbic",
    "Hippocampus_R": "Limbic",
    "ParaHippocampal_L": "Limbic",
    "ParaHippocampal_R": "Limbic",
    "Amygdala_L": "Limbic",
    "Amygdala_R": "Limbic",
    "Temporal_Pole_Sup_L": "Limbic",
    "Temporal_Pole_Sup_R": "Limbic",

    # -------------------------
    # Subcortical / 기타
    # -------------------------
    "Caudate_L": "Other",
    "Caudate_R": "Other",
    "Putamen_L": "Other",
    "Putamen_R": "Other",
    "Pallidum_L": "Other",
    "Pallidum_R": "Other",
    "Thalamus_L": "Other",
    "Thalamus_R": "Other",
    "Temporal_Mid_L": "Other",
    "Temporal_Mid_R": "Other",
    "Temporal_Inf_L": "Other",
    "Temporal_Inf_R": "Other",
    "Temporal_Pole_Mid_L": "Other",
    "Temporal_Pole_Mid_R": "Other",
    "Rectus_L": "Other",
    "Rectus_R": "Other",
    "Cingulum_Mid_L": "Other",
    "Cingulum_Mid_R": "Other",
}


def get_network(name):
    if name not in ROI_TO_NETWORK:
        print(f"⚠️ Unknown ROI: {name}")
    return ROI_TO_NETWORK.get(name, "Other")


def make_network_txt(input_txt, output_txt):

    with open(input_txt, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    net_counter = Counter()

    with open(output_txt, "w") as f:
        for line in lines:
            roi = line.split()[0]

            net = get_network(roi)
            net_counter[net] += 1

            f.write(f"{roi} {net}\n")

    print(f"\nSaved → {output_txt}")
    print(f"Total ROIs: {sum(net_counter.values())}")

    print("\nNetwork distribution:")
    for net, cnt in net_counter.items():
        print(f"{net:15s}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="aal_network_labels.txt")

    args = parser.parse_args()

    make_network_txt(args.input, args.output)