import scanpy as sc
import pandas as pd
import mygene
import json
import requests

TARGET_SIZE = 768
vendor_overlap = ['6440', '3861', '6401', '1511', '2315', '50616', '8490', '3310', '1311', '6363', '3553', '6358', '6362', '3605', '3002', '6364', '3820', '5105', '2920', '3001', '1215', '8076', '8857', '3569', '8115', '6361', '2206', '3458', '8685', '2069', '4489', '1791', '9173', '2205', '51237', '10631', '931', '6355', '10563', '6357', '924', '112744', '338', '3003', '50489', '6370', '2219', '2162', '8530', '1308', '3596', '4283', '54474', '6356', '916', '4318', '4760', '973', '914', '11075', '3491', '3824', '3627', '3162', '890', '83888', '1805', '11326', '2999', '429', '1490', '5179', '5055', '6369', '177', '6367']
lineage = ['7852', '2212', '151887', '3728', '2022', '1513', '1289', '10203', '10320', '2260', '7048', '1290', '5159', '6515', '83483', '3898', '873', '1525', '9308', '3689', '302', '55384', '3587', '6711', '2013', '780', '947', '55303', '2034', '7431', '4072', '1956', '1718', '9332', '3852', '960', '6688', '597', '7077', '4131', '4240', '64123', '7040', '5156', '84525', '5175', '1284', '968', '6383', '871', '3977', '9076', '7070', '409', '6387', '4360', '476', '23327', '11167', '1003', '5792', '224', '7122', '1366', '161198', '8436', '7102', '3075', '7035', '5788', '22918', '1535', '5268', '3561', '58494']

super_classes = {
    "mechanical": {
        "tissues": ["Skin", "Lung", "Esophagus", "Colon", "Duodenum", "Rectum", "Small intestine", "Cervix"],
        "quotas": {
            "structural_geometry": {
                "panel_weight": int(TARGET_SIZE * 0.30),
                "go_id": [
                    "GO:0045111",  # Intermediate filament (Rescues all the missing Keratins and Vimentin)
                    "GO:0005884",  # Actin filament (Rescues the contractile/microvilli mechanics)
                    "GO:0005874",  # Microtubule (Rescues the Tubulins for cilia/shape)
                    "GO:0005856",  # Cytoskeleton (A broader net to catch associated anchor proteins)
                    "GO:0005198"   # parent node
                ]
            },
            "adhesion_matrix": {
                "panel_weight": int(TARGET_SIZE * 0.25),
                "go_id": [
                    "GO:0030057",  # Desmosome (Crucial for identifying skin/squamous shear-stress layers)
                    "GO:0005912",  # Adherens junction (Catches the classic Cadherins for sheet integrity)
                    "GO:0005925",  # Focal adhesion (Catches the Integrins attaching cells to the ECM/floor)
                    "GO:0030056",  # Hemidesmosome (The absolute anchor for Basal Keratinocytes)
                    "GO:0005911"   # cell junction
                ]
            },
            "epithelial_barrier": {
                "panel_weight": int(TARGET_SIZE * 0.15),
                "go_id": ["GO:0008544", "GO:0005923", "GO:0005929", "GO:0005902"]
            },
            "turnover_stratification": {
                "panel_weight": int(TARGET_SIZE * 0.10),
                "go_id": [
                    "GO:0000278",  # Mitotic cell cycle (Catches active G1/S and G2/M phases)
                    "GO:0008283",  # Cell population proliferation (Catches growth rate vectors)
                    "GO:0042981",  # Regulation of apoptotic process (Catches cell death trajectories)
                    "GO:0090398"   # cellular senescence
                ]
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


def fetch_go_direct(go_ids):
    # Base endpoint for the BioThings / MyGene API
    url = "https://mygene.info/v3/query"
    gene_list = []
    page_total = 1
    skip_count = 0
    total_genes_available = 0

    for gID in go_ids:
        print(f"Executing direct GET request for {gID}...")
        while skip_count < page_total:
            print(f"skip count: {skip_count}, page total: {page_total}")
            params = {
                "q": gID,
                "fields": "entrezgene",  # Stops it from returning "everything"
                # "scopes": "go",
                "size": 1000,  # Overrides the default 10-result limit
                "skip": skip_count
            }
            if gID != "GO:0007568":
                params["species"] = "human"

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                # print(data)
                page_total = data.get('total', 0)
                skip_count += 1000

                for hit in data["hits"]:
                    gene_list.append(hit["entrezgene"])
            else:
                print(f"❌ API Request Failed. Status Code: {response.status_code}")
                print(response.text)
                return []
        total_genes_available += page_total
        skip_count = 0
        page_total = 1
    print(f"✅ Success: Retrieved {len(gene_list)} genes (out of {total_genes_available} total available).")
    return list(set(gene_list))




def build_deduplicated_panel(adata_pan_raw, config):
    print("Initiating Sequential Bucket Fill...")

    # The master set that guarantees uniqueness
    flat_panel = set(vendor_overlap)
    hr_panel = {
        "mechanical": {
            "lineage": [],
            "vendor_overlap": vendor_overlap,
            "structural_geometry": [],
            "adhesion_matrix": [],
            "turnover_stratification": [],
            "epithelial_barrier": [],
            "spillover": []
        }
    }

    lineage_added = 0
    for gene in lineage:
        if gene not in flat_panel:
            flat_panel.add(gene)
            hr_panel["mechanical"]["lineage"].append(gene)
            lineage_added += 1

    print(f"Added {lineage_added} unique Lineage genes. Total: {len(flat_panel)}, {len(hr_panel["mechanical"]["lineage"])}")

    # NEW: Create a pristine pool of leftovers to use instead of random HVGs
    spillover_pool = set()

    for query_type, query_details in config["quotas"].items():
        # print(f"getting go genes for {query_type} from {query_details['go_id']}")
        candidate_genes = fetch_go_direct(query_details["go_id"])
        # print(f"got {len(candidate_genes)} genes")

        valid_candidates = [g for g in candidate_genes if g in adata_pan_raw.var_names]
        adata_bucket = adata_pan_raw[:, valid_candidates].copy()

        print("determining HVGs")
        sc.pp.highly_variable_genes(
            adata_bucket,
            # layer="counts",  # Use the raw integer layer
            flavor='seurat_v3',
            batch_key="batch_tissue",  # CRITICAL: Prevents organ-specific bias
            n_top_genes=len(valid_candidates)
        )

        print("sort them bitches")
        ranked_genes = adata_bucket.var.sort_values(by='variances_norm', ascending=False).index.tolist()

        print("add genes to panel")
        added_count = 0
        for gene in ranked_genes:
            if gene not in flat_panel:
                if added_count < query_details["panel_weight"]:
                    # Fill the primary quota
                    hr_panel["mechanical"][query_type].append(gene)
                    flat_panel.add(gene)
                    added_count += 1
                else:
                    # The bucket is full, save this highly-ranked gene for emergencies
                    spillover_pool.add(gene)

        print(hr_panel)

    # 5. THE CASCADE SPILLOVER (Replacing the HVG padding)
    final_list = list(flat_panel)
    if len(final_list) < TARGET_SIZE:
        missing = TARGET_SIZE - len(final_list)
        print(f"Buckets under-filled by {missing}. Engaging Cascade Spillover...")

        # Rank the spillover pool by overall variance so we grab the best leftovers
        adata_spill = adata_pan_raw[:, list(spillover_pool)].copy()

        sc.pp.highly_variable_genes(
            adata_spill,
            # layer="counts",  # Use the raw integer layer
            flavor='seurat_v3',
            batch_key="batch_tissue",  # CRITICAL: Prevents organ-specific bias
            n_top_genes=len(spillover_pool)
        )

        ranked_spillover = adata_spill.var.sort_values(by='variances_norm', ascending=False).index.tolist()

        for gene in ranked_spillover:
            if len(final_list) == TARGET_SIZE:
                break
            if gene not in final_list:
                hr_panel["mechanical"]["spillover"].append(gene)
                final_list.append(gene)


    print(f"✅ Panel Generation Complete. Final Size: {len(final_list)}")

    with open("../Tensors/MechanicalGenePanel.json", 'w') as oFile:
        json.dump(hr_panel, oFile)

    return final_list

# available_gene_pool = find_hardware_overlap("../Cellular Data/Single Cell/Reference/10x5kPanel.csv",
#                                             "../Cellular Data/Single Cell/Reference/nano6kPanel.csv")


# print(fetch_go_direct(["GO:0007049"]))

input_file = "raw_mechanical.h5ad"

print("Opening ", input_file)
adata = sc.read_h5ad("../Cellular Data/Single Cell/Reference/" + input_file)

print(build_deduplicated_panel(adata, super_classes["mechanical"]))






















