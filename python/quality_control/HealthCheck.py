import json

import numpy
import requests
import scanpy as sc
import gc
import numpy as np
# import pandas as pd
# import mygene
from scipy.sparse import issparse

# 1. Load your spatial map (The "Rosetta Stone")
# This ensures we can translate standard gene symbols into the Entrez IDs you need.
print("Loading spatial map...")
with open('../cell_type_model_builder/symbol2entrez.json', 'r') as f:
    symbol_to_id = json.load(f)

file_types = [
      "lung",
      # "Gut",
      # "Vasculature",
      # "skin"
]
back_spatial_genes = [
    "100133941",
    "1003",
    "100506658",
    "101",
    "10252",
    "1026",
    "10266",
    "1027",
    "1029",
    "10298",
    "1030",
    "10398",
    "10409",
    "10413",
    "1043",
    "10462",
    "10516",
    "10538",
    "10578",
    "1063",
    "10630",
    "10631",
    "1066",
    "10715",
    "10894",
    "11065",
    "113146",
    "114548",
    "114569",
    "114795",
    "117159",
    "1178",
    "1236",
    "125981",
    "1277",
    "1282",
    "1290",
    "1291",
    "1294",
    "1303",
    "1308",
    "1359",
    "1363",
    "1364",
    "1365",
    "1366",
    "137075",
    "1381",
    "1382",
    "1407",
    "1408",
    "140885",
    "1410",
    "142",
    "1432",
    "1462",
    "1464",
    "147495",
    "1493",
    "1521",
    "1522",
    "1525",
    "1535",
    "1545",
    "155465",
    "158326",
    "1592",
    "1594",
    "1645",
    "168537",
    "1690",
    "171024",
    "171177",
    "1717",
    "1728",
    "1823",
    "1828",
    "1831",
    "1844",
    "1869",
    "1956",
    "1958",
    "1959",
    "196",
    "199",
    "1991",
    "2006",
    "207",
    "208",
    "2167",
    "2171",
    "218",
    "2199",
    "219931",
    "2200",
    "2201",
    "2205",
    "2207",
    "22809",
    "22915",
    "22918",
    "22933",
    "22949",
    "2312",
    "2315",
    "2331",
    "2335",
    "23406",
    "23410",
    "23411",
    "23705",
    "240",
    "241",
    "2487",
    "255738",
    "25778",
    "25797",
    "25928",
    "259307",
    "26018",
    "260429",
    "2624",
    "2625",
    "26291",
    "2669",
    "2706",
    "27076",
    "27134",
    "2729",
    "2829",
    "283420",
    "283652",
    "28513",
    "28514",
    "28984",
    "28996",
    "29108",
    "2919",
    "2947",
    "29956",
    "3002",
    "3003",
    "30835",
    "3146",
    "3148",
    "3151",
    "3161",
    "3164",
    "3217",
    "3280",
    "330",
    "3308",
    "3312",
    "332",
    "3371",
    "3394",
    "3398",
    "340348",
    "340665",
    "3429",
    "3458",
    "3485",
    "3486",
    "3490",
    "3552",
    "3553",
    "3559",
    "3563",
    "3569",
    "358",
    "3589",
    "360",
    "3606",
    "3620",
    "3624",
    "3638",
    "3655",
    "3659",
    "3662",
    "3670",
    "3687",
    "3688",
    "3713",
    "3718",
    "374",
    "374897",
    "3815",
    "3820",
    "3849",
    "3852",
    "3856",
    "3866",
    "3875",
    "387763",
    "3880",
    "3909",
    "3910",
    "3911",
    "3912",
    "3913",
    "3914",
    "3915",
    "3918",
    "3925",
    "3932",
    "3953",
    "4000",
    "4001",
    "4050",
    "4052",
    "4053",
    "4054",
    "4060",
    "4069",
    "4070",
    "4088",
    "4089",
    "4157",
    "4239",
    "4254",
    "4256",
    "4286",
    "4288",
    "4308",
    "4312",
    "4313",
    "4314",
    "4316",
    "4317",
    "4318",
    "4319",
    "4320",
    "4321",
    "4322",
    "4323",
    "4327",
    "4332",
    "4345",
    "4353",
    "441027",
    "445",
    "448834",
    "4494",
    "4501",
    "4609",
    "4615",
    "4629",
    "467",
    "468",
    "4684",
    "472",
    "474",
    "4780",
    "4783",
    "4818",
    "4854",
    "4862",
    "4929",
    "4968",
    "50489",
    "5055",
    "5058",
    "50848",
    "50943",
    "5099",
    "51062",
    "51203",
    "51237",
    "51458",
    "51548",
    "5156",
    "5159",
    "51629",
    "5167",
    "5187",
    "5268",
    "5347",
    "5366",
    "54474",
    "545",
    "54504",
    "5467",
    "54829",
    "5499",
    "55076",
    "5518",
    "55365",
    "5551",
    "5553",
    "55740",
    "558",
    "5595",
    "5629",
    "5654",
    "56603",
    "572",
    "5730",
    "57326",
    "5734",
    "57402",
    "5742",
    "5743",
    "57496",
    "5764",
    "57705",
    "5790",
    "58484",
    "59",
    "5914",
    "5915",
    "5916",
    "5925",
    "59352",
    "5950",
    "5996",
    "6035",
    "6036",
    "608",
    "6095",
    "6096",
    "6097",
    "6241",
    "6256",
    "6257",
    "6258",
    "6279",
    "6280",
    "6284",
    "6285",
    "633",
    "634",
    "6352",
    "6363",
    "6367",
    "6374",
    "63827",
    "6387",
    "64066",
    "6422",
    "64220",
    "6423",
    "642587",
    "644672",
    "64499",
    "6490",
    "6507",
    "6515",
    "6546",
    "6590",
    "6624",
    "6647",
    "6648",
    "6649",
    "6657",
    "6662",
    "6663",
    "6665",
    "667",
    "6678",
    "6772",
    "6775",
    "6790",
    "6925",
    "7020",
    "7021",
    "7026",
    "7037",
    "7040",
    "7042",
    "7046",
    "7048",
    "7058",
    "7070",
    "7076",
    "7077",
    "7078",
    "7079",
    "7082",
    "712",
    "7122",
    "713",
    "7133",
    "7153",
    "7157",
    "7177",
    "7227",
    "728",
    "7295",
    "7298",
    "7299",
    "7305",
    "7351",
    "7421",
    "7450",
    "7477",
    "7852",
    "79094",
    "79148",
    "79365",
    "79630",
    "7980",
    "79924",
    "800",
    "8074",
    "8076",
    "80781",
    "8091",
    "834",
    "83483",
    "836",
    "8404",
    "841",
    "8425",
    "847",
    "8490",
    "8553",
    "8581",
    "8626",
    "863",
    "8678",
    "8842",
    "8848",
    "8864",
    "891",
    "9069",
    "9076",
    "90865",
    "909",
    "90993",
    "910",
    "911",
    "914",
    "915",
    "91522",
    "916",
    "917",
    "9173",
    "920",
    "9232",
    "9235",
    "924",
    "925",
    "926",
    "929",
    "930",
    "9308",
    "9332",
    "941",
    "9414",
    "9415",
    "9423",
    "9437",
    "9447",
    "9452",
    "947",
    "9515",
    "9572",
    "9575",
    "958",
    "959",
    "960",
    "9641",
    "968",
    "969",
    "973",
    "9804",
    "983",
    "991",
    "995",
    "9975",
    "998",
    "9982",
    "999"
]
spatial_gene_symbols = ["AARS", "ACER1", "ACTA2", "ADAM8", "ADM2", "AGR3", "AHNAK2", "AHR", "AIF1", "AIM2", "AKR1C1", "AKT1", "AKT2", "ALDH3A1", "ALOX5", "ALOX5AP", "APCDD1", "AQP1", "AQP3", "AREG", "ARNTL", "ASPN", "ASS1", "ATF3", "ATF4", "ATF5", "ATL1", "ATM", "ATOH1", "ATR", "AURKA", "AXL", "BAD", "BASP1", "BATF", "BCAN", "BECN1", "BGN", "BHLHE40", "BHLHE41", "BIRC3", "BIRC5", "C11orf96", "C1QA", "C1QB", "C1orf54", "C5AR1", "CADM1", "CALD1", "CASP1", "CASP3", "CASP8", "CAT", "CBFA2T3", "CCL19", "CCL22", "CCL5", "CCNB1", "CCR7", "CD14", "CD163", "CD19", "CD1A", "CD1B", "CD1C", "CD2", "CD200", "CD207", "CD209", "CD24", "CD34", "CD3D", "CD3E", "CD3G", "CD4", "CD40", "CD40LG", "CD44", "CD52", "CD68", "CD69", "CD7", "CD79A", "CD80", "CD83", "CD8A", "CD8B", "CD93", "CDC20", "CDC25C", "CDC42", "CDH1", "CDH19", "CDH5", "CDK1", "CDKN1A", "CDKN1B", "CDKN2A", "CDKN2B", "CEACAM1", "CENPF", "CERS1", "CERS2", "CES1", "CHAC1", "CLC", "CLDN1", "CLDN12", "CLDN23", "CLDN25", "CLDN3", "CLDN4", "CLDN5", "CLDN7", "CLEC10A", "CLEC9A", "CLOCK", "COCH", "COL12A1", "COL17A1", "COL18A1", "COL1A1", "COL23A1", "COL4A1", "COL5A2", "COL6A1", "COL7A1", "COTL1", "CPA3", "CPE", "CPVL", "CRABP1", "CRABP2", "CREB3L1", "CRY1", "CRY2", "CRYAB", "CSPG4", "CTLA4", "CTSW", "CTSZ", "CXADR", "CXCL1", "CXCL12", "CXCL5", "CXCR4", "CYBA", "CYP1B1", "CYP26A1", "CYP26B1", "CYP26C1", "CYP27B1", "DCD", "DHCR7", "DLL1", "DSC1", "DSG1", "DST", "DSTYK", "DUSP2", "E2F1", "EGFR", "EGR1", "EGR2", "ELANE", "ELN", "ENAH", "ENPP1", "F11R", "FABP4", "FABP5", "FADS2", "FBLN2", "FBLN5", "FBN1", "FBN2", "FCER1A", "FCER1G", "FGF21", "FGF23", "FGFBP1", "FLG", "FMOD", "FN1", "FOXP3", "FREM1", "FRZB", "FSCN1", "GATA2", "GATA3", "GCLC", "GEM", "GIMAP7", "GJB2", "GNLY", "GSTM3", "GZMB", "GZMK", "H2AFX", "HES1", "HIPK2", "HIST1H4C", "HMGA2", "HMGB1", "HMGB2", "HMGN2", "HMMR", "HOXB7", "HSPA4", "HSPA8", "HTRA1", "ID2", "IDO1", "IFI27", "IFNG", "IGFBP2", "IGFBP3", "IGFBP7", "IKBKE", "IL11", "IL18", "IL1A", "IL1B", "IL1RL1", "IL2RA", "IL32", "IL33", "IL3RA", "IL4I1", "IL6", "INHBA", "INSIG1", "IRF1", "IRF4", "IRF8", "ISL1", "ITGA6", "ITGAX", "ITGB1", "ITM2A", "IVL", "JAK3", "KIT", "KITLG", "KLRB1", "KPRP", "KRT15", "KRT18", "KRT19", "KRT2", "KRT20", "KRT5", "KRT8", "LAMA3", "LAMA4", "LAMA5", "LAMB1", "LAMB2", "LAMB3", "LAMC1", "LAMC2", "LCK", "LEPR", "LGR6", "LMNA", "LMNB1", "LOR", "LRIG1", "LTB", "LTBP1", "LTBP2", "LTBP3", "LTBP4", "LUM", "LY6D", "LYPD3", "LYVE1", "LYZ", "MAL2", "MAPK14", "MAPK3", "MC1R", "MFAP4", "MFAP5", "MGP", "MIR205HG", "MITF", "MKI67", "MLANA", "MMP1", "MMP10", "MMP11", "MMP12", "MMP13", "MMP14", "MMP19", "MMP2", "MMP27", "MMP28", "MMP3", "MMP7", "MMP8", "MMP9", "MMRN1", "MNDA", "MPO", "MRTFB", "MT1F", "MT1X", "MYC", "MYD88", "MYH11", "MYL9", "MZB1", "NCAM1", "NCR1", "NDUFA4L2", "NFE2L2", "NFIL3", "NKG7", "NLRC4", "NLRP3", "NOTCH3", "NPAS2", "NQO1", "NR1D1", "NR1D2", "NR2F2", "NR4A1", "NR4A2", "NTN1", "NUSAP1", "OCLN", "OGG1", "PAK1", "PAK4", "PARP1", "PBXIP1", "PCDH7", "PCSK9", "PDGFRA", "PDGFRB", "PDPN", "PER1", "PER2", "PLK1", "PLVAP", "PMAIP1", "PMEL", "POSTN", "PPARD", "PPP1CA", "PPP2R1A", "PRF1", "PRG2", "PROM1", "PROX1", "PRSS33", "PTGDS", "PTGER4", "PTGR1", "PTGS1", "PTGS2", "PTN", "PTPRCAP", "PTTG1", "PYCARD", "QPCT", "RAMP2", "RARA", "RARB", "RARG", "RB1", "RBP4", "RGCC", "RGS1", "RGS5", "RHCG", "RHOV", "RNASE1", "RNASE2", "RORA", "RORB", "RORC", "RRM2", "RXRA", "RXRB", "RXRG", "S100A13", "S100A14", "S100A8", "S100A9", "S100B", "SBSN", "SERPINB2", "SERPINB5", "SFRP1", "SFRP2", "SIRPA", "SIRT1", "SIRT2", "SIRT3", "SIRT6", "SLC1A3", "SLC24A5", "SLC25A39", "SLC2A3", "SLC8A1", "SLPI", "SMAD3", "SMAD4", "SOD1", "SOD2", "SOD3", "SOSTDC1", "SOX10", "SOX15", "SOX2", "SOX9", "SPARC", "SPARCL1", "SPRY1", "STAT1", "STAT4", "STMN1", "STRA6", "STXBP5L", "SYNPO2", "TACSTD2", "TCF4", "TFAP2A", "TFAP2B", "TFPI2", "TFRC", "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "THBS2", "THY1", "TIMP1", "TIMP2", "TIMP3", "TIMP4", "TJP1", "TJP2", "TJP3", "TMEM132B", "TMEM150C", "TMEM176A", "TMEM45A", "TNC", "TNFRSF17", "TNFRSF1B", "TOMM20", "TOP2A", "TP53", "TP63", "TPCN2", "TPSAB1", "TPSB2", "TRPM1", "TRPS1", "TSC22D1", "TSC22D3", "TSPAN33", "TXN", "TYMS", "TYR", "TYROBP", "UBE2C", "UCP2", "VCAN", "VDR", "VWF", "WDFY4", "WNT7B", "XCR1", "YAP1"]

for file_type in file_types:

    # Define paths
    # input_path = f"../Cellular Data/Single Cell/Reference/raw{file_type}.h5ad"
    input_path = f"../../cellular_data/Single Cell/Reference/HSCA_extended.h5ad"
    output_path = f"../../clean_skin.h5ad"
    print(f"\n--- Processing Skin ---")
    adata = sc.read_h5ad(input_path)

    # 1. Check for "Dead" cells (High Mitochondrial content)
    # High mito % usually means the cell membrane ruptured, leaving only junk RNA.
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

    # 2. Filter out the noise
    print(f"Cells before QC: {adata.n_obs}")
    # Keep cells with at least 200 genes and < 20% mitochondrial RNA
    adata = adata[adata.obs.n_genes_by_counts > 200, :]
    adata = adata[adata.obs.pct_counts_mt < 20, :]
    print(f"Cells after QC: {adata.n_obs}")

    # 3. Handle duplicates
    print("Enforcing unique names...")
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # 4. Final Sanity Check for scANVI (OOM-Safe)
    print("Calculating sparsity...")
    if issparse(adata.X):
        # .nnz is a built-in attribute that uses zero extra memory
        total_elements = adata.n_obs * adata.n_vars
        sparsity = 100 * (1.0 - (adata.X.nnz / total_elements))
    else:
        # Fallback if it's already a dense numpy array
        sparsity = 100 * (1.0 - (np.count_nonzero(adata.X) / adata.X.size))

    print(f"Final Count Matrix Sparsity: {sparsity:.2f}%")

    print(adata.var_names[:5])

    # Calculate the mean total counts per cell
    mean_depth = adata.X.sum(axis=1).mean()
    print(f"True Training Target Depth: {mean_depth}")

    print("Training shape", adata.shape)

    # 1. Look at the raw matrix values
    # sample_data = adata.X[:5, :5].toarray() if issparse(adata.X) else adata.X[:5, :5]
    # print("Matrix Sample:\n", sample_data)
    #
    # # 2. Check the maximum value
    # max_val = adata.X.max()
    # print(f"Max Expression Value: {max_val}")
    #
    # # max_val = raw_adata.raw.max()
    # # print(f"Max Raw Expression Value: {max_val}")
    #
    # # 3. Check for fractions/decimals
    # is_integer = numpy.all(numpy.equal(numpy.mod(sample_data, 1), 0))
    # print(f"Are all values raw integers? {is_integer}")

    # break

    present_genes = [g for g in spatial_gene_symbols if g in adata.var_names]

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    query_data = {
        'q': ','.join(adata.var_names),
        'scopes': 'symbol',
        'fields': 'entrezgene',
        'species': 'human'
    }

    response = requests.post("https://mygene.info/v3/query", data=query_data, headers=headers)
    response.raise_for_status()
    mapping_results = response.json()

    # 1. Initialize collections
    symbol_to_entrez = {}
    dropped_symbols = []
    kept_symbols = []

    # 2. Iterate through the results
    for result in mapping_results:
        symbol = result.get('query')

        # Check if a mapping was actually found
        if 'entrezgene' in result:
            # Some symbols might return multiple hits; we'll take the first one
            # or you can add logic here to choose the best one
            symbol_to_entrez[symbol] = int(result['entrezgene'])
            kept_symbols.append(symbol)
        else:
            dropped_symbols.append(symbol)

    # 3. Print the report
    print(f"📊 Mapping Summary:")
    print(f"   - Total Queried: {len(adata.var_names)}")
    print(f"   - Successfully Mapped: {len(kept_symbols)}")
    print(f"   - Dropped: {len(dropped_symbols)}")

    if dropped_symbols:
        print("\n❌ Dropped Genes List:")
        for s in dropped_symbols:
            print(f"   - {s}")

    # --- NEW: RESCUE THE RAW COUNTS ---
    print("Rescuing raw integers from layers['counts']...")
    if 'counts' in adata.layers:
        # Overwrite the log-normalized floats with the raw integers
        adata.X = adata.layers['counts'].copy()
    else:
        raise ValueError("CRITICAL: 'counts' layer missing! Cannot proceed without raw data.")

    # Now it is safe to nuke the metadata to save memory
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]
    for key in list(adata.layers.keys()):
        del adata.layers[key]

    gc.collect()

    print("Building new smaller dataset from overlap...")
    adata_shrunk = adata[:, kept_symbols].copy()

    del adata
    gc.collect()

    # --- 5. VERIFY INTEGER COUNTS ---
    # We DELETE the np.round() logic because rounding is a hack.
    # We just verify that the rescue worked.
    print("Verifying counts are true integers...")
    sample_val = adata_shrunk.X.data[0] if issparse(adata_shrunk.X) else adata_shrunk.X[0, 0]
    print(f"Sample data type: {type(sample_val)}, Value: {sample_val}")

    print(f"✅ Sliced successfully. New shape: {adata_shrunk.shape}")

    # --- 6. SAVE AND NUKE ---
    print(f"Writing slim file to disk: {output_path}")
    adata_shrunk.write_h5ad(output_path, compression="gzip")

    del adata_shrunk
    gc.collect()

print("\n🎉 All reference files successfully cleaned, translated, and shrunk!")