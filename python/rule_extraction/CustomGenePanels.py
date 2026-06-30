import scanpy as sc
import json

TARGET_SIZE = 768
vendor_overlap = ['6440', '3861', '6401', '1511', '2315', '50616', '8490', '3310', '1311', '6363', '3553', '6358', '6362', '3605', '3002', '6364', '3820', '5105', '2920', '3001', '1215', '8076', '8857', '3569', '8115', '6361', '2206', '3458', '8685', '2069', '4489', '1791', '9173', '2205', '51237', '10631', '931', '6355', '10563', '6357', '924', '112744', '338', '3003', '50489', '6370', '2219', '2162', '8530', '1308', '3596', '4283', '54474', '6356', '916', '4318', '4760', '973', '914', '11075', '3491', '3824', '3627', '3162', '890', '83888', '1805', '11326', '2999', '429', '1490', '5179', '5055', '6369', '177', '6367']
lineage = ['7852', '2212', '151887', '3728', '2022', '1513', '1289', '10203', '10320', '2260', '7048', '1290', '5159', '6515', '83483', '3898', '873', '1525', '9308', '3689', '302', '55384', '3587', '6711', '2013', '780', '947', '55303', '2034', '7431', '4072', '1956', '1718', '9332', '3852', '960', '6688', '597', '7077', '4131', '4240', '64123', '7040', '5156', '84525', '5175', '1284', '968', '6383', '871', '3977', '9076', '7070', '409', '6387', '4360', '476', '23327', '11167', '1003', '5792', '224', '7122', '1366', '161198', '8436', '7102', '3075', '7035', '5788', '22918', '1535', '5268', '3561', '58494']

# 1. Define the Super-Classes and their specific Tissues
super_classes = {
    "mechanical": {
        "tissues": ["Skin", "Lung", "Esophagus", "Colon", "Duodenum", "Rectum", "Small intestine", "Cervix"],
        "quotas": {
            "structural_core": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": "GO:0005200"
            },
            "adhesion": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": "GO:0005911"
            },
            "turnover_aging": {
                "panel_weight": int(TARGET_SIZE * 0.20),
                "go_id": "GO:0007049"
            },
            "inflammatory_sasp": {
                "panel_weight": int(TARGET_SIZE * 0.10),
                "go_id": "GO:0006954"
            }
        }
    },
    "metabolic": {
        "tissues": ["Pancreas", "Liver", "Salivary gland", "Gallbladder", "Prostate"],
        "quotas": {
            "secretion_exocytosis": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": "GO:0046903"
            },
            "metabolic": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": "GO:0044281"
            },
            "turnover_aging": {
                "panel_weight": int(TARGET_SIZE * 0.20),
                "go_id": "GO:0007049"
            },
            "inflammatory_sasp": {
                "panel_weight": int(TARGET_SIZE * 0.10),
                "go_id": "GO:0006954"
            }
        }
    },
    "endocrine": {
        "tissues": ["Adrenal", "Parathyroid", "Ovary", "Endometrium", "Placenta"],
        "quotas": {
            "hormone": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": "GO:0005179"
            },
            "vascular": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": "GO:0001525"
            },
            "turnover_aging": {
                "panel_weight": int(TARGET_SIZE * 0.20),
                "go_id": "GO:0007049"
            },
            "inflammatory_sasp": {
                "panel_weight": int(TARGET_SIZE * 0.10),
                "go_id": "GO:0006954"
            }
        }
    },
}
hr_panel = {
        "mechanical": {
            "lineage": [],
            "vendor_overlap": vendor_overlap,
            "structural_core": [],
            "adhesion": [],
            "turnover_aging": [],
            "inflammatory_sasp": []
        }
    }


# config = super_classes["mechanical"]
#
# for query_type, query_details in config["quotas"].items():
#     print("query type: ", query_type)
#     print("go id: ", query_details["go_id"])
#     print("weight: ", query_details["panel_weight"])

import mygene


def get_go_genes(go_term, target_format="entrezgene"):
    mg = mygene.MyGeneInfo()
    # Query the GO term, filter for Human (species=9606)
    results = mg.query(f"go:{go_term}", species="human", fields=target_format, fetch_all=True)

    # Extract the requested IDs (e.g., Entrez IDs)
    gene_list = []
    for hit in results:
        if target_format in hit:
            # Handle cases where one gene has multiple Entrez IDs
            if isinstance(hit[target_format], list):
                gene_list.extend([str(x) for x in hit[target_format]])
            else:
                gene_list.append(str(hit[target_format]))

    return list(set(gene_list))  # Remove duplicates


# 2. The Main Generator Engine
def generate_super_panel(adata_full, class_name, config):
    print(f"\n--- Building {class_name.upper()} Panel ---")

    # Slice the massive atlas down to JUST the target tissues for this model
    adata_sub = adata_full[adata_full.obs['tissue_type'].isin(config["tissues"])].copy()

    # Start with the hardware agnostic intersection (10%) + Lineage Anchors (10%)
    panel = set(vendor_overlap + lineage)

    panel = {class_name: {
        "lineage"
    }}

    # Fill the biological buckets dynamically
    # for go_term, slot_count in config["quotas"].items():
    for query_type, query_details in config["quotas"].items():
        candidate_genes = get_go_genes(query_details["go_id"])

        # Filter for genes that actually exist in your dataset
        valid_candidates = [g for g in candidate_genes if g in adata_sub.var_names]
        adata_bucket = adata_sub[:, valid_candidates].copy()

        # Let the math find the highest variance genes for THIS specific tissue group
        sc.pp.highly_variable_genes(adata_bucket, n_top_genes=query_details["panel_weight"], batch_key="tissue_type")
        best_genes = adata_bucket.var[adata_bucket.var['highly_variable']].index.tolist()

        panel.update(best_genes)
        print(f"Filled bucket with {len(best_genes)} genes.")

    # Convert to list and enforce the strict WebGPU constraint
    final_panel = list(panel)[:TARGET_SIZE]

    # Safety Check: If the math didn't find enough, pad with general HVGs
    if len(final_panel) < TARGET_SIZE:
        print("Warning: Buckets under-filled. Padding with general highly variable genes.")
        sc.pp.highly_variable_genes(adata_sub, n_top_genes=TARGET_SIZE, batch_key="tissue_type")
        general_hvgs = adata_sub.var[adata_sub.var['highly_variable']].index.tolist()
        for gene in general_hvgs:
            if len(final_panel) >= TARGET_SIZE: break
            if gene not in final_panel: final_panel.append(gene)

    # Save the specific panel for the Rust Router
    filename = f"model_genes_{class_name}.json"
    with open(filename, "w") as f:
        json.dump(final_panel, f)

    print(f"✅ Saved {filename} ({len(final_panel)} genes)")
    return final_panel

# 3. Execute the generator (assuming 'adata_pan' contains all 25 tissues)
# generate_super_panel(adata_pan, "mechanical", super_classes["mechanical"])
# panel_meta = generate_super_panel(adata_pan, "metabolic", super_classes["metabolic"])
# panel_endo = generate_super_panel(adata_pan, "endocrine", super_classes["endocrine"])