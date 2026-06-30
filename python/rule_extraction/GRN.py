# Created by Ryan Berg March 18th, 2026
# Purpose: Extract the transcriptomic physics from UV-Protectd young skin
# Extraction parameter: noise,

def main():

    # import sdevelo
    # import scvelo
    # import scanpy
    #
    # import pandas
    # import numpy
    #
    # import json
    #
    # print("reading annotated data...")
    # skin25 = scanpy.read_h5ad("../Cell Data/GSE130973/SRR9036396/counts_unfiltered/annSkin25.h5ad") # cache=True
    #
    # print("updating layer names")
    # # Map the kb-python naming to scVelo naming
    # skin25.layers['spliced'] = skin25.layers['mature']
    # skin25.layers['unspliced'] = skin25.layers['nascent']
    #
    # # (Optional) Remove the old keys to save memory
    # del skin25.layers['mature']
    # del skin25.layers['nascent']
    #
    # print("filter and normalize")
    # # 1. Filter genes based on counts
    # scvelo.pp.filter_genes(skin25, min_shared_counts=20)
    #
    # # 2. Normalize and Log (Standard Scanpy functions)
    # scanpy.pp.normalize_total(skin25)
    # scanpy.pp.log1p(skin25)
    #
    # # 3. Select Highly Variable Genes (Replacement for 'filter_genes_dispersion')
    # # This is where your n_top_genes=2000 belongs
    # scanpy.pp.highly_variable_genes(skin25, n_top_genes=2000)
    #
    # # 4. Subset to the selected genes
    # skin25 = skin25[:, skin25.var.highly_variable].copy()
    #
    # # 5. Dimensionality Reduction
    # scanpy.tl.pca(skin25)
    # scanpy.pp.neighbors(skin25, n_pcs=30, n_neighbors=30)
    #
    # # 6. Apply the PAGA Bugfix (Required for v0.3.x)
    # # This resolves the "unexpected format" warning you saw
    # skin25.uns['neighbors']['distances'] = skin25.obsp['distances']
    # skin25.uns['neighbors']['connectivities'] = skin25.obsp['connectivities']
    #
    # print("Distances in uns:", 'distances' in skin25.uns['neighbors'])
    # print("Connectivities in uns:", 'connectivities' in skin25.uns['neighbors'])
    #
    # # 7. Compute Moments
    # scvelo.pp.moments(skin25)
    #
    # # 8. Recover Dynamics (with your increased core count)
    # scvelo.tl.recover_dynamics(skin25, n_jobs=8)
    # print("compute velocity")
    # scvelo.tl.velocity(skin25, mode="dynamical")
    #
    # # Weighted adjacency matrix (Casual edges for each gene, aka W matrix)
    # print("compute casual edges")
    # scvelo.tl.velocity_graph(skin25)
    #
    # # gene-specific noise tensor
    # sdeConfig = sdevelo.Config()
    # sdeConfig.nEpochs = 2
    # sdeConfig.batchSz = 64
    # sdeConfig.scv_n_jobs = 8
    # sdeConfig.n_gene = 2000
    #
    # print("init sde model")
    # model = sdevelo.SDENN(sdeConfig, skin25)
    #
    # print("Train...🚂...cho cho motherfucker...")
    # model.train(sdeConfig.nEpochs)

    import sdevelo
    import scvelo
    import scanpy
    import json

    print("reading annotated data...")
    adata = scanpy.read_h5ad("../Cell Data/GSE130973/SRR9036396/counts_unfiltered/annSkin25.h5ad")
    # Manually inject the 'seed' attribute into the AnnData object

    print("updating layer names")
    # Map the kb-python naming to scVelo naming
    adata.layers['spliced'] = adata.layers['mature']
    adata.layers['unspliced'] = adata.layers['nascent']

    # (Optional) Remove the old keys to save memory
    del adata.layers['mature']
    del adata.layers['nascent']

    # 1. Manual Preprocessing (Bypasses the error)
    # Use your 'nac' output adata here
    print("filter and normalize")
    scanpy.pp.filter_cells(adata, min_genes=200)
    scanpy.pp.filter_genes(adata, min_cells=3)
    scanpy.pp.normalize_total(adata, target_sum=1e4)  # Modern replacement
    scanpy.pp.log1p(adata)
    scanpy.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()

    print("pca")
    scanpy.tl.pca(adata)

    # 2. Compute Neighbors (The slow part - keep n_neighbors low)
    # n_neighbors=30 is standard; don't go higher on 24GB RAM
    print("neighbors")
    scanpy.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

    # 3. Compute Moments (This creates 'Ms' and 'Mu')
    print("moments...........")
    scvelo.pp.moments(adata)

    # 2. Compute velocity first (SDEvelo needs the initial velocity vectors)
    print("fast velocity")
    scvelo.tl.velocity(adata, mode='stochastic')

    # 3. Initialize and Fit SDENN
    # We pass 'preprocessed=True' (if available) or just ensured adata is ready
    print("init model")
    args = sdevelo.Config()
    args.process = False
    args.sde_mode = "torchsde"

    model = sdevelo.SDENN(args, adata)

    print("SDENN Initialized. Proceed with training.")
    print("train.. cho cho")
    model.train(2)  # Keep epochs low for a quick noise check

    genes = adata.var_names.tolist()

    def get_param(param):
        return param.detach().cpu().numpy().tolist()

    print("Extracting univariate SDE parameters...")

    physics_payload = {
        "genes": genes,
        "model_type": "univariate_kinetics",
        "parameters": {
            "a": get_param(model.a),  # Activation time shift
            "b": get_param(model.b),  # Activation steepness
            "c": get_param(model.c),  # Max transcription rate
            "beta": get_param(model.beta),  # Splicing rate
            "gamma": get_param(model.gamma),  # Degradation rate
            "sigma1": get_param(model.sigma1),  # Unspliced Noise
            "sigma2": get_param(model.sigma2)  # Spliced Noise
        }
    }

    with open("sde_physics_constants.json", "w") as f:
        json.dump(physics_payload, f, indent=4)

    print("SUCCESS: Biological engine physics extracted.")

    # genes = adata.var_names.tolist()
    #
    # # Extract Kinetic Rates
    # # beta = splicing rate, gamma = degradation rate
    # kinetics = {
    #     "beta": model.beta.detach().cpu().numpy().tolist(),
    #     "gamma": model.gamma.detach().cpu().numpy().tolist()
    # }
    #
    # # The 'Aging Engine' Payload
    # physics_payload = {
    #     "genes": genes,
    #     "diffusion_noise_sigma": noise_vector.tolist(),
    #     "kinetics": kinetics,
    #     "jacobian_wiring": {
    #         genes[i]: {
    #             genes[j]: float(jacobian_2d[i, j])
    #             for j in range(len(genes)) if abs(jacobian_2d[i, j]) > 1e-4
    #         }
    #         for i in range(len(genes))
    #     }
    # }
    #
    # with open("skin_sde_physics.json", "w") as f:
    #     json.dump(physics_payload, f, indent=4)
    #
    # print("SUCCESS: Phase 1, Step 2 is fully complete.")



    # # 3. The Crowbar: Let's see what it modified or holds
    # print("\n--- NEW ADATA VAR COLUMNS ---")
    # # scVelo usually prepends 'velocity' or 'fit'. Let's see what SDEvelo added.
    # print(adata.var.columns.tolist())
    #
    # print("\n--- NEW ADATA LAYERS ---")
    # # Did it output a new layer for the noise?
    # print(adata.layers.keys())
    #
    # print("\n--- MODEL METHODS & ATTRIBUTES ---")
    # # Let's find the actual extraction function
    # print([attr for attr in dir(model) if not attr.startswith('_')])
    #
    # # 4. Extract the Noise (Diffusion) Parameter
    # # This is usually stored as 'sigma' in the model's learned parameters
    # noise_param = model.sigma.detach().cpu().numpy()
    # print(f"Estimated Noise Parameter (Sigma): {noise_param}")

if __name__ == "__main__":
    main()

# model.

# import sdevelo
# config = sdevelo.Config()
#
# # Print all available parameters (filtering out built-in methods)
# print([attr for attr in dir(config) if not attr.startswith('_')])
#
# # Alternatively, check for a built-in dictionary
# print(config.__dict__)



























