# import scipy.sparse as sparse

# cellTargets = [
#     'Basal KC',
#     'Prolif. KC',
#     'Spinous KC',
#     'Granular KC',
#     'Cornified KC'
# ]
#
# print("Reading Full AnnData")
# hscaExtended = scanpy.read_h5ad("../Single Cell Data/Reference/HSCA_extended.h5ad")
#
# if hscaExtended.raw is not None:
#     del hscaExtended.raw
# if 'distances' in hscaExtended.obsp:
#     del hscaExtended.obsp['distances']
# if 'connectivities' in hscaExtended.obsp:
#     del hscaExtended.obsp['connectivities']
#
# # Filter Reference Data
# print("Filtering")
# hscaExtended = hscaExtended[
#     (~hscaExtended.obs["age_years"].isna()) &
#     (hscaExtended.obs["age_years"] != 0.0) &
#     (hscaExtended.obs["Condition"] == "Healthy") &
#     (hscaExtended.obs["celltype_lvl_3_extended"].isin(cellTargets))
# ].copy()
#
# print("Compress to sparse matrix")
# hscaExtended.X = sparse.csr_matrix(hscaExtended.X)
#
# print("Saving...")
# hscaExtended.write_h5ad("../Single Cell Data/Reference/Filtered_HSCA_extended.h5ad", compression="gzip")