import json
import warnings

import numpy as np
import scanpy as sc
import scvi
import torch
import scipy.sparse as sp
import mygene

from sklearn.metrics import classification_report, accuracy_score
import gc

gene_panel = ['6355', '7408', '130271', '7077', '5818', '2264', '7187', '5328', '3605', '2890', '4360', '3695', '6926',
              '8842', '11035', '27130', '5914', '8857', '118', '1000', '977', '823', '84557', '347733', '5420',
              '203068', '10892', '1284', '28514', '5792', '1019', '182', '3977', '55303', '5795', '23385', '6383',
              '8744', '1803', '667', '4887', '3852', '9314', '6714', '6622', '161198', '11075', '4058', '5027', '6362',
              '8321', '4643', '8766', '5584', '8490', '3925', '924', '3491', '22900', '10203', '1029', '10544', '1511',
              '3693', '2322', '22861', '1812', '154796', '7402', '3718', '4853', '10694', '1499', '27429', '7412',
              '26136', '6370', '6361', '7122', '54474', '1718', '1756', '7297', '10788', '364', '2735', '3716', '238',
              '7157', '5315', '1012', '1501', '2737', '4926', '3339', '5583', '2885', '7185', '8412', '8837', '1969',
              '22918', '5604', '1612', '10576', '23239', '834', '488', '3799', '843', '7852', '80216', '6369', '2261',
              '5747', '6708', '2314', '5423', '1525', '7525', '11011', '873', '914', '5296', '2252', '10320', '2904',
              '8626', '2296', '1641', '84687', '5754', '58494', '4602', '5339', '2219', '968', '8795', '7414', '2268',
              '10076', '65018', '8826', '598', '9620', '64123', '1308', '8295', '4644', '5599', '931', '1813', '7048',
              '3001', '4771', '11315', '1366', '2923', '10631', '5590', '6363', '2534', '966', '8841', '6515', '1830',
              '156', '9053', '1294', '960', '1739', '634', '7409', '1952', '79444', '4921', '8322', '3861', '55081',
              '1601', '841', '83700', '23411', '4646', '7010', '4040', '6387', '7046', '5175', '286', '2999', '1445',
              '3796', '83888', '9223', '3596', '5788', '50507', '2206', '83593', '102', '3673', '5594', '25937', '3587',
              '8754', '5783', '816', '3162', '3897', '6416', '1003', '5582', '9564', '8436', '5268', '10253', '11186',
              '10411', '5179', '50489', '7189', '81', '50848', '2185', '3635', '1948', '9076', '3832', '7102', '127602',
              '2258', '429', '51237', '81631', '3002', '597', '799', '4642', '5329', '3363', '5376', '5587', '5105',
              '3694', '22806', '9266', '4323', '581', '2034', '5371', '7159', '3611', '6440', '89797', '8874', '4254',
              '10392', '8076', '29108', '9332', '50616', '1213', '8607', '284', '9112', '1464', '89796', '10533',
              '3702', '3676', '8650', '9019', '387', '4217', '10763', '1378', '2319', '4706', '7277', '7791', '3169',
              '9308', '23327', '8767', '56288', '9448', '6662', '1311', '5055', '29109', '8727', '112744', '5580',
              '860', '224', '64236', '11326', '3690', '11076', '25', '6711', '4137', '25932', '1285', '4851', '8829',
              '3815', '5154', '1030', '317', '25816', '890', '3932', '2012', '24145', '6401', '10382', '6772', '23542',
              '1791', '84707', '4131', '2212', '2852', '7534', '7070', '80149', '1808', '3685', '1436', '6840', '472',
              '4134', '6357', '23136', '3918', '9173', '5170', '51741', '27', '3159', '1063', '7454', '4162', '7188',
              '858', '23513', '2069', '9212', '375', '3482', '4599', '5797', '2920', '5781', '3728', '8976', '2242',
              '3458', '2022', '171024', '382', '4072', '3383', '867', '4067', '177', '408', '79648', '3824', '976',
              '5777', '1073', '659', '120892', '1909', '25833', '3561', '3480', '6367', '87', '27185', '8773', '2205',
              '3674', '3075', '6616', '3569', '3313', '5819', '3691', '2013', '4283', '4690', '1490', '7283', '1432',
              '3553', '5156', '3636', '57142', '10397', '5817', '6237', '596', '1289', '84617', '3898', '3479', '1495',
              '4747', '7317', '2801', '9817', '6868', '2043', '10783', '4478', '1656', '3055', '3643', '5595', '5467',
              '79633', '3678', '7052', '6358', '409', '2017', '908', '10413', '11004', '5499', '5581', '6396', '1859',
              '2260', '3655', '2162', '2274', '2010', '9414', '1027', '4179', '1778', '4288', '947', '1639', '6464',
              '186', '2318', '64127', '6514', '6513', '207', '7465', '1215', '9475', '55384', '3310', '1535', '151887',
              '7074', '10309', '5602', '595', '5295', '51203', '10563', '22974', '4291', '3689', '1452', '2697', '8530',
              '6688', '6356', '308', '637', '11167', '4267', '25945', '1896', '2241', '2317', '1759', '3820', '84525',
              '10128', '4430', '602', '4804', '5297', '3384', '4627', '83483', '5034', '1020', '7535', '4489', '59272',
              '7082', '7422', '7248', '3627', '10603', '3312', '2113', '10243', '5600', '8115', '332', '1513', '214',
              '7280', '10525', '4318', '1956', '1540', '7431', '4092', '10015', '3717', '604', '5347', '916', '7035',
              '1290', '10013', '5028', '3572', '5530', '4760', '3672', '84433', '53340', '10856', '9833', '6790',
              '7040', '3845', '1009', '8685', '10018', '1785', '22933', '6753', '780', '4240', '2315', '56171', '6364',
              '9902', '4908', '5727', '23603', '973', '871', '3003', '5159', '51174', '476', '3675', '3362', '3914',
              '7486', '10971', '6093', '10970', '1805', '4218', '302', '1861', '338', '1385', '995', '64170', '468',
              '1017', '842', '578', '5601', '8452', '329', '9126', '1025', '8451', '8915', '331', '3661', '1616',
              '3981', '7161', '6722', '328', '1642', '1398', '86', '8454', '1994', '2064', '7849', '6500', '4605',
              '10524', '23435', '7917', '9978', '599', '8772', '5515', '3265', '993', '7332', '5367', '7057', '1440',
              '919', '948', '5054', '925', '6863', '27306', '929', '399', '6374', '10225', '366', '1906', '2199', '917',
              '23166', '6754', '728', '4481', '2350', '9021', '25890', '8519', '7124', '9023', '6402', '6775', '926',
              '2214', '864', '6368', '6271', '1236', '29851', '9839', '5327', '1282', '51348', '3248', '5142', '8406',
              '1441', '3456', '1545', '4321', '6424', '1437', '10232', '2624', '9452', '313', '4973', '3586', '6422',
              '56729', '10663', '6571', '9935', '5579', '51655', '4017', '1910', '10417', '1536', '5743', '794', '7056',
              '3164', '6236', '3624', '8821', '201633', '1493', '11009', '8013', '4052', '3481', '9201', '123', '6446',
              '8809', '467', '639', '7153', '640', '911', '55784', '10538', '10231', '23670', '7058', '2122', '6556',
              '64321', '4208', '959', '56253', '54504', '7037', '9518', '11065', '290', '3558', '10462', '3953', '1604',
              '6653', '10516', '7133', '3037', '3604', '57007', '23569', '4094', '939', '5336', '5243', '54', '133',
              '5360', '6354', '913', '7049', '1075', '23236', '25928', '1088', '3687', '4982', '3516', '2357', '91319',
              '3400', '3904', '5168', '6373', '5649', '5793', '5657', '117157', '11126', '608', '9760', '5629', '3560',
              '7294', '2078', '30817', '3557', '4790', '8320']
clean_panel = [str(g) for g in gene_panel]

import requests # Use this instead of mygene

# ==========================================
# 1. Load the Configuration & Hardware Limits
# ==========================================
print("1. Loading JSON artifacts...")
version = "4"
folder_path = f"../Cell Classifier/v{version}"

# A. Xenium Genes (541 physically visible genes)
with open("../Cellular Data/Spatial/Manchester/back/gene_panel.json", "r") as f:
    panel_data = json.load(f)
targets = panel_data.get('payload', {}).get('targets', panel_data)
xenium_symbols = [t['type']['data']['name'] for t in targets if t.get('type', {}).get('descriptor') == 'gene']

# DEBUG: Prove we actually extracted symbols
print(f"   - Extracted {len(xenium_symbols)} symbols from Xenium JSON.")

# B. Map Xenium Symbols -> Entrez IDs (Bypassing hishel bug)
print("   - Mapping Xenium Symbols to Entrez IDs via raw API call...")
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
query_data = {
    'q': ','.join(xenium_symbols),
    'scopes': 'symbol',
    'fields': 'entrezgene',
    'species': 'human'
}

response = requests.post("https://mygene.info/v3/query", data=query_data, headers=headers)
response.raise_for_status()
mapping_results = response.json()

xenium_entrez_set = set()
for item in mapping_results:
    if 'entrezgene' in item:
        xenium_entrez_set.add(str(item['entrezgene']))

print(f"   - Successfully mapped {len(xenium_entrez_set)} Xenium genes to Entrez IDs.")

# B. Model Genes & Means (The 768 genes the model expects)
with open(f"{folder_path}/model_genes.json", "r") as f:
    model_genes = json.load(f)

with open(f"{folder_path}/model_means.json", "r") as f:
    model_means = np.array(json.load(f))

# C. Architecture Config
with open(f"{folder_path}/model_config.json", "r") as f:
    config = json.load(f)["non_kwargs"]

# ==========================================
# 2. Resurrect the Model & Reference Data
# ==========================================
print("2. Resurrecting Model and Data...")
raw_adata = sc.read_h5ad("../Cellular Data/Single Cell/Reference/sc_mech.h5ad")

print("copying raw counts to counts layer")
raw_adata.layers["counts"] = raw_adata.raw.X.copy()

print("reducing reference to 768 gene panel")
adata_subset = raw_adata[:, clean_panel].copy()
print("sending out unsliced reference for garbage collection")
del raw_adata
gc.collect()

# Setup SCVI pointing specifically to the raw integer counts layer
scvi.model.SCVI.setup_anndata(
    adata_subset,
    layer="counts", # Force scvi to use the raw counts, ignoring .X
    batch_key="batch_tissue"
)

# Build the SCVI skeleton
scvi_model = scvi.model.SCVI(
    adata_subset,
    n_hidden=config.get("n_hidden", 128),
    n_latent=config.get("n_latent", 30),
    n_layers=config.get("n_layers", 2)
)

# Load the weights into the SCVI skeleton (Bypass PyTorch 2.6 security check)
checkpoint = torch.load(f"{folder_path}/model.pt", map_location='cpu', weights_only=False)
state_dict = checkpoint["model_state_dict"]

# strict=False ignores the unexpected pyro keys
scvi_model.module.load_state_dict(state_dict, strict=False)
scvi_model.is_trained_ = True

# UPGRADE to SCANVI to attach the Classifier head
model = scvi.model.SCANVI.from_scvi_model(
    scvi_model,
    labels_key="mechanical_cell_type",
    unlabeled_category="Unknown"
)

model.is_trained_ = True
print("   - Model successfully reconstructed and upgraded to SCANVI.")

# ==========================================
# 2.5 SANITY CHECK: The "Upper Bound" Test
# ==========================================
import random

print("\n--- RUNNING GUT CHECK ON KNOWN DATA ---")

# 1. Grab 5 random cells from your un-blinded reference data
random_indices = random.sample(range(adata_subset.n_obs), 5)
gut_check_adata = adata_subset[random_indices].copy()

# 2. Get predictions AND confidence scores
# soft=True returns the dataframe of probabilities for all 4 classes
probs = model.predict(gut_check_adata, soft=True)
preds = model.predict(gut_check_adata)

# 3. Compare with Ground Truth
true_labels = gut_check_adata.obs["mechanical_cell_type"].values

for i in range(5):
    # Handle pandas series vs numpy array returns
    predicted_class = preds.iloc[i] if hasattr(preds, 'iloc') else preds[i]
    true_class = true_labels[i]

    # Get the confidence percentage for the chosen class
    confidence = probs.iloc[i][predicted_class] * 100

    match = "✅" if predicted_class == true_class else "❌"
    print(f"{match} True: {true_class:<15} | Pred: {predicted_class:<15} | Conf: {confidence:.1f}%")

print("---------------------------------------\n")

# ==========================================
# 3. Memory-Efficient Xenium Simulation
# ==========================================
print("3. Blinding the 768-gene model (Memory Optimized & Entrez Aligned)...")

# 1. Identify which indices need to be imputed
# model_genes are now your Entrez IDs (e.g., "10413")
model_genes = adata_subset.var_names.tolist()

# THE FUSION: We intersect against the Entrez set generated in Step 1
overlapping_entrez = set(model_genes).intersection(xenium_entrez_set)

# Get the column indices of the genes NOT in the physical Xenium panel
impute_indices = [i for i, gene in enumerate(model_genes) if gene not in overlapping_entrez]
print(f"   - Extracted {len(xenium_symbols)} gene symbols from Xenium JSON.") # ADD THIS LINE
print(f"   - TRUE Overlap: {len(overlapping_entrez)} real. Imputing {len(impute_indices)} genes.")


# 2. Extract the counts layer (avoiding full adata copy)
counts = adata_subset.layers['counts']
if not sp.isspmatrix_csr(counts):
    print("   - Converting to CSR format for efficient indexing...")
    counts = counts.tocsr()

# 3. Create the simulated matrix in float32 (Half the memory of float64)
print(f"   - Allocating float32 matrix (~5.2GB for 1.8M x 768)...")
sim_matrix = counts.astype(np.float32).toarray()

# 4. Vectorized Imputation
print("   - Injecting baseline means...")
# This one-liner instantly overwrites all 1.8 million cells for the missing genes
sim_matrix[:, impute_indices] = model_means[impute_indices]

# 5. Re-wrap into a light Anndata for prediction
# We don't copy the Great Merge's heavy metadata; we just swap the matrix
adata_sim = sc.AnnData(X=sim_matrix, obs=adata_subset.obs, var=adata_subset.var)
adata_sim.layers['counts'] = sim_matrix # scvi looks here
# print(f"   - Successfully imputed {missing_count} missing genes using model_means.json.")

# ==========================================
# 4. Final Performance Audit (Direct Handshake)
# ==========================================
print("4. Testing classification accuracy...")

# We filter the 'unnormalized count' warning because we are using
# float-based model_means for the simulation.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*does not contain unnormalized count data.*")

    # IMPORTANT: Pass adata_sim as the first argument.
    # scvi will detect it's a new object and perform a 'transfer_setup' automatically.
    preds = model.predict(adata_sim, batch_size=2048)

# Grab the ground truth from your subsetted reference
y_true = adata_subset.obs["mechanical_cell_type"]

# Ensure labels match (Categorical alignment)
y_pred = preds.values if hasattr(preds, 'values') else preds

print(f"\n" + "=" * 30)
print(f"✅ ABLATION AUDIT COMPLETE")
print(f"=" * 30)
print(f"Genes in Model:    {len(model_genes)}")
print(f"Genes in Xenium:   {len(overlapping_entrez)}") # FIXED VARIABLE NAME
print(f"Imputed (Blinded): {len(model_genes) - len(overlapping_entrez)}") # FIXED VARIABLE NAME
print(f"Simulated Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"=" * 30)

print("\n--- Detailed Classification Report ---")
print(classification_report(y_true, y_pred))