from torch.nn.functional import dropout
# Force MPI to only look at the local machine and ignore network cards (en0)
import os
os.environ['OMPI_MCA_btl'] = 'self,sm,vader'
os.environ['MPICH_INTERFACE_HOSTNAME'] = 'localhost'

def main():
    # import onnx
    from sklearn.metrics import classification_report
    import logging
    import pandas as pd
    # import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import numpy
    import requests
    import torch
    from safetensors.torch import save_file
    import scipy.sparse as sciSparse

    import scanpy
    import gc
    import json
    import scvi
    from pytorch_lightning.loggers import TensorBoardLogger
    import torch.nn as nn

    import os
    # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    scvi.settings.verbosity = logging.INFO
    scvi.settings.progress_bar_style = "rich"
    scvi.settings.dl_num_workers = 0
    # scvi.settings.dl_persistent_workers = True

    number_of_genes = 473

    # print(torch.__version__)
    logger = TensorBoardLogger("triplicate_logs", name="IDTrainer")
    version = 20
    print(f"making directory to store output in Cell Classifier/v{version}")
    os.makedirs(f"../H&E Computer Vision/RNA_Cell_Classification/models/v{version}", exist_ok=True)

    # Execution:
    # num_broad = len(broad_scanvi_model.adata_manager.get_state_registry("labels").categorical_mapping)
    # num_granular = len(granular_scanvi_model.adata_manager.get_state_registry("labels").categorical_mapping)
    # dual_model = DualHeadSCANVI(input_genes=473, num_broad_classes=num_broad, num_granular_classes=num_granular)
    # dual_model.load_state_dict(extract_dual_head_weights(broad_scanvi_model, granular_scanvi_model), strict=True)

    def extract_dual_head_weights(broad_model, granular_model, num_broad_wrapper, num_granular_wrapper):
        # broad_sd = broad_model.module.state_dict()
        # granular_sd = granular_model.module.state_dict()

        new_sd = {}

        # --- 1. SHARED Z-ENCODER ---
        new_sd['z_encoder_l0.weight'] = granular_model['z_encoder.encoder.fc_layers.Layer 0.0.weight']
        new_sd['z_encoder_l0.bias'] = granular_model['z_encoder.encoder.fc_layers.Layer 0.0.bias']
        new_sd['z_encoder_bn0.weight'] = granular_model['z_encoder.encoder.fc_layers.Layer 0.1.weight']
        new_sd['z_encoder_bn0.bias'] = granular_model['z_encoder.encoder.fc_layers.Layer 0.1.bias']
        new_sd['z_encoder_bn0.running_mean'] = granular_model['z_encoder.encoder.fc_layers.Layer 0.1.running_mean']
        new_sd['z_encoder_bn0.running_var'] = granular_model['z_encoder.encoder.fc_layers.Layer 0.1.running_var']

        new_sd['z_encoder_l1.weight'] = granular_model['z_encoder.encoder.fc_layers.Layer 1.0.weight']
        new_sd['z_encoder_l1.bias'] = granular_model['z_encoder.encoder.fc_layers.Layer 1.0.bias']
        new_sd['z_encoder_bn1.weight'] = granular_model['z_encoder.encoder.fc_layers.Layer 1.1.weight']
        new_sd['z_encoder_bn1.bias'] = granular_model['z_encoder.encoder.fc_layers.Layer 1.1.bias']
        new_sd['z_encoder_bn1.running_mean'] = granular_model['z_encoder.encoder.fc_layers.Layer 1.1.running_mean']
        new_sd['z_encoder_bn1.running_var'] = granular_model['z_encoder.encoder.fc_layers.Layer 1.1.running_var']

        new_sd['z_mean_encoder.weight'] = granular_model['z_encoder.mean_encoder.weight']
        new_sd['z_mean_encoder.bias'] = granular_model['z_encoder.mean_encoder.bias']

        # --- 2. BROAD CLASSIFIER (With Padding) ---
        new_sd['broad_l0.weight'] = broad_model['classifier.classifier.0.fc_layers.Layer 0.0.weight']
        new_sd['broad_l0.bias'] = broad_model['classifier.classifier.0.fc_layers.Layer 0.0.bias']
        new_sd['broad_bn0.weight'] = broad_model['classifier.classifier.0.fc_layers.Layer 0.1.weight']
        new_sd['broad_bn0.bias'] = broad_model['classifier.classifier.0.fc_layers.Layer 0.1.bias']
        new_sd['broad_bn0.running_mean'] = broad_model['classifier.classifier.0.fc_layers.Layer 0.1.running_mean']
        new_sd['broad_bn0.running_var'] = broad_model['classifier.classifier.0.fc_layers.Layer 0.1.running_var']

        new_sd['broad_l1.weight'] = broad_model['classifier.classifier.0.fc_layers.Layer 1.0.weight']
        new_sd['broad_l1.bias'] = broad_model['classifier.classifier.0.fc_layers.Layer 1.0.bias']
        new_sd['broad_bn1.weight'] = broad_model['classifier.classifier.0.fc_layers.Layer 1.1.weight']
        new_sd['broad_bn1.bias'] = broad_model['classifier.classifier.0.fc_layers.Layer 1.1.bias']
        new_sd['broad_bn1.running_mean'] = broad_model['classifier.classifier.0.fc_layers.Layer 1.1.running_mean']
        new_sd['broad_bn1.running_var'] = broad_model['classifier.classifier.0.fc_layers.Layer 1.1.running_var']

        b_weight = broad_model['classifier.classifier.1.weight']
        b_bias = broad_model['classifier.classifier.1.bias']

        # Pad to 5 classes if it's 4
        if b_weight.shape[0] < num_broad_wrapper:
            pad_size = num_broad_wrapper - b_weight.shape[0]
            b_weight = torch.cat([b_weight, torch.zeros(pad_size, b_weight.shape[1])], dim=0)
            b_bias = torch.cat([b_bias, torch.zeros(pad_size)], dim=0)

        new_sd['broad_out.weight'] = b_weight
        new_sd['broad_out.bias'] = b_bias

        # --- 3. GRANULAR CLASSIFIER (With Padding) ---
        new_sd['granular_l0.weight'] = granular_model['classifier.classifier.0.fc_layers.Layer 0.0.weight']
        new_sd['granular_l0.bias'] = granular_model['classifier.classifier.0.fc_layers.Layer 0.0.bias']
        new_sd['granular_bn0.weight'] = granular_model['classifier.classifier.0.fc_layers.Layer 0.1.weight']
        new_sd['granular_bn0.bias'] = granular_model['classifier.classifier.0.fc_layers.Layer 0.1.bias']
        new_sd['granular_bn0.running_mean'] = granular_model['classifier.classifier.0.fc_layers.Layer 0.1.running_mean']
        new_sd['granular_bn0.running_var'] = granular_model['classifier.classifier.0.fc_layers.Layer 0.1.running_var']

        new_sd['granular_l1.weight'] = granular_model['classifier.classifier.0.fc_layers.Layer 1.0.weight']
        new_sd['granular_l1.bias'] = granular_model['classifier.classifier.0.fc_layers.Layer 1.0.bias']
        new_sd['granular_bn1.weight'] = granular_model['classifier.classifier.0.fc_layers.Layer 1.1.weight']
        new_sd['granular_bn1.bias'] = granular_model['classifier.classifier.0.fc_layers.Layer 1.1.bias']
        new_sd['granular_bn1.running_mean'] = granular_model['classifier.classifier.0.fc_layers.Layer 1.1.running_mean']
        new_sd['granular_bn1.running_var'] = granular_model['classifier.classifier.0.fc_layers.Layer 1.1.running_var']

        g_weight = granular_model['classifier.classifier.1.weight']
        g_bias = granular_model['classifier.classifier.1.bias']

        # Pad to 25 classes if it's 24
        if g_weight.shape[0] < num_granular_wrapper:
            pad_size = num_granular_wrapper - g_weight.shape[0]
            g_weight = torch.cat([g_weight, torch.zeros(pad_size, g_weight.shape[1])], dim=0)
            g_bias = torch.cat([g_bias, torch.zeros(pad_size)], dim=0)

        new_sd['granular_out.weight'] = g_weight
        new_sd['granular_out.bias'] = g_bias

        return new_sd

    class DualHeadSCANVI(nn.Module):
        def __init__(self, input_genes: int, num_broad_classes: int, num_granular_classes: int, latent_dim: int = 30):
            super().__init__()

            # --- 1. Z-ENCODER (Shared Scaffold) ---
            self.z_encoder_l0 = nn.Linear(input_genes, 128)
            self.z_encoder_bn0 = nn.BatchNorm1d(128)

            self.z_encoder_l1 = nn.Linear(128, 128)
            self.z_encoder_bn1 = nn.BatchNorm1d(128)

            self.z_mean_encoder = nn.Linear(128, latent_dim)

            # --- 2. BROAD CLASSIFIER (Head A) ---
            self.broad_l0 = nn.Linear(latent_dim, 128)
            self.broad_bn0 = nn.BatchNorm1d(128)

            self.broad_l1 = nn.Linear(128, 128)
            self.broad_bn1 = nn.BatchNorm1d(128)

            self.broad_out = nn.Linear(128, num_broad_classes)

            # --- 3. GRANULAR CLASSIFIER (Head B) ---
            self.granular_l0 = nn.Linear(latent_dim, 128)
            self.granular_bn0 = nn.BatchNorm1d(128)

            self.granular_l1 = nn.Linear(128, 128)
            self.granular_bn1 = nn.BatchNorm1d(128)

            self.granular_out = nn.Linear(128, num_granular_classes)

            self.relu = nn.ReLU()

        def forward(self, x):
            # Apply the log1p transform natively
            x = torch.log1p(x)

            # Z-Encoder
            x = self.relu(self.z_encoder_bn0(self.z_encoder_l0(x)))
            x = self.relu(self.z_encoder_bn1(self.z_encoder_l1(x)))
            z = self.z_mean_encoder(x)

            # Broad Branch
            b = self.relu(self.broad_bn0(self.broad_l0(z)))
            b = self.relu(self.broad_bn1(self.broad_l1(b)))
            logits_broad = self.broad_out(b)

            # Granular Branch
            g = self.relu(self.granular_bn0(self.granular_l0(z)))
            g = self.relu(self.granular_bn1(self.granular_l1(g)))
            logits_granular = self.granular_out(g)

            # The model returns 3 tensors: Latent coordinates, Broad Logits, Granular Logits
            return z, logits_broad, logits_granular

    # ==========================================
    # 1. TRAINING PHASE
    # ==========================================

    print("reading raw mechanical reference")
    raw_adata = scanpy.read_h5ad("../Cellular Data/Single Cell/Reference/skin_only_mech.h5ad")

    print(raw_adata.var_names[:5])

    # Calculate the mean total counts per cell
    mean_depth = raw_adata.X.sum(axis=1).mean()
    print(f"True Training Target Depth: {mean_depth}")

    print("Training shape", raw_adata.shape)

    # 1. Look at the raw matrix values
    sample_data = raw_adata.X[:5, :5].toarray() if sciSparse.issparse(raw_adata.X) else raw_adata.X[:5, :5]
    print("Matrix Sample:\n", sample_data)

    # 2. Check the maximum value
    max_val = raw_adata.X.max()
    print(f"Max Expression Value: {max_val}")

    # 3. Check for fractions/decimals
    is_integer = numpy.all(numpy.equal(numpy.mod(sample_data, 1), 0))
    print(f"Are all values raw integers? {is_integer}")

    bad_labels = ['Unknown', 'unknown', 'nan', 'NaN', 'None']

    # 1. Mask for Mechanical (Convert to string to catch literal 'nan' strings safely)
    mech_valid = ~raw_adata.obs['mechanical_cell_type'].astype(str).isin(bad_labels)

    # 2. Mask for Granular (Check strings AND check for actual pd.NA/np.nan objects)
    gran_valid = ~raw_adata.obs['granular_cell_type'].astype(str).isin(bad_labels) & raw_adata.obs[
        'granular_cell_type'].notna()

    # 3. Apply the combined mask
    adata_tensor = raw_adata[mech_valid & gran_valid].copy()

    print(f"🧹 Strict Filter Applied. Cells retained: {adata_tensor.n_obs}")
    del raw_adata
    gc.collect()

    final_mean_depth = float(adata_tensor.X.sum(axis=1).mean())
    print(f"📏 Final Training Target Depth: {final_mean_depth:.4f}")

    # 1. Verification of the Purge
    # class_counts = adata_tensor.obs['mechanical_cell_type'].value_counts()
    # print("--- Final Class Distribution ---")
    # print(class_counts)

    adata_tensor.obs['mechanical_cell_type'] = adata_tensor.obs['mechanical_cell_type'].astype('category')
    adata_tensor.obs['granular_cell_type'] = adata_tensor.obs['granular_cell_type'].astype('category')

    # 2. Explicitly inject "Unknown" into the allowed categories if it was filtered out
    if "Unknown" not in adata_tensor.obs['mechanical_cell_type'].cat.categories:
        adata_tensor.obs['mechanical_cell_type'] = adata_tensor.obs['mechanical_cell_type'].cat.add_categories(
            ["Unknown"])

    if "Unknown" not in adata_tensor.obs['granular_cell_type'].cat.categories:
        adata_tensor.obs['granular_cell_type'] = adata_tensor.obs['granular_cell_type'].cat.add_categories(["Unknown"])

    #
    #
    # print("setting up model")
    # # Setup AnnData with the Batch Key (Crucial for Pan-Tissue)
    # scvi.model.SCVI.setup_anndata(
    #     adata_tensor,
    #     # layer="counts",
    #     labels_key='mechanical_cell_type',
    #     batch_key='batch_tissue' # e.g., "Skin", "Lung", "Gut"
    # )
    # torch.set_num_threads(8)
    #
    # # Train the Unsupervised Latent Space (scVI)
    # print("Training scVI Latent Space...")
    # pan_scvi_model = scvi.model.SCVI(adata_tensor, n_layers=2, n_latent=30, gene_likelihood="nb",)
    # pan_scvi_model.train(accelerator="mps", devices=1)
    # torch.set_num_threads(8)
    #
    # # Train the Supervised Classifier (scANVI)
    # print("Training scANVI Classifier...")
    # pan_scanvi_model = scvi.model.SCANVI.from_scvi_model(
    #     pan_scvi_model,
    #     adata = adata_tensor,
    #     unlabeled_category="Unknown",
    #     dropout_rate=0.2
    # )
    #
    # try:
    #     pan_scanvi_model.train(
    #         max_epochs=100,
    #         batch_size=2048,
    #         accelerator="mps",
    #         devices=1,
    #         # Trainer-level parameters
    #         early_stopping=True,
    #         early_stopping_patience=4,
    #         early_stopping_min_delta=0.001,
    #         check_val_every_n_epoch=1,
    #         logger=logger,
    #         # Optimization parameters (The "Plan")
    #         plan_kwargs={
    #             "lr": 5e-4,
    #             "weight_decay": 1e-4,
    #             "reduce_lr_on_plateau": True, # Optional but recommended
    #         }
    #     )
    # except KeyboardInterrupt:
    #     print("ending training early. moving to save")
    # except Exception as ex:
    #     print(ex)
    #
    # print("✅ Training Complete. Starting Wasm Export Pipeline...")

    torch.set_num_threads(8)

    # ==========================================
    # PHASE 1: Unsupervised Latent Space (scVI)
    # ==========================================
    # 100
    print("Training scVI Latent Space (Base)...")
    # Notice we DO NOT pass a labels_key here. We only care about batches/tissues.
    scvi.model.SCVI.setup_anndata(
        adata_tensor,
        batch_key='batch_tissue'
    )

    pan_scvi_model = scvi.model.SCVI(adata_tensor, n_layers=2, n_latent=30, gene_likelihood="nb")
    pan_scvi_model.train(accelerator="mps", devices=1, max_epochs=100, early_stopping=True, logger=logger)

    # ==========================================
    # PHASE 2: Structural / Broad Classifier
    # ==========================================
    # 30
    print("Training Broad scANVI Classifier...")
    # We re-setup the AnnData to point to the mechanical labels
    scvi.model.SCANVI.setup_anndata(
        adata_tensor,
        labels_key='mechanical_cell_type',
        unlabeled_category="Unknown"
    )

    broad_scanvi_model = scvi.model.SCANVI.from_scvi_model(
        pan_scvi_model,
        labels_key='mechanical_cell_type',  # Add this
        unlabeled_category="Unknown",  # Add this
        adata=adata_tensor,
        dropout_rate=0.2
    )

    broad_scanvi_model.train(
        max_epochs=30,  # Usually needs fewer epochs because latent space is already trained
        batch_size=2048,
        accelerator="mps",
        devices=1,
        early_stopping=True,
        early_stopping_patience=4,
        early_stopping_min_delta=0.001,
        plan_kwargs={"lr": 5e-4, "weight_decay": 1e-4, "reduce_lr_on_plateau": True},
        logger=logger
    )

    # ==========================================
    # PHASE 3: Granular Classifier
    # ==========================================
    # 50
    print("Training Granular scANVI Classifier...")
    # Re-setup the AnnData for the granular labels
    scvi.model.SCANVI.setup_anndata(
        adata_tensor,
        labels_key='granular_cell_type',
        unlabeled_category="Unknown"
    )

    granular_scanvi_model = scvi.model.SCANVI.from_scvi_model(
        pan_scvi_model,
        labels_key='granular_cell_type',
        unlabeled_category="Unknown",
        adata=adata_tensor,
        dropout_rate=0.2
    )

    granular_scanvi_model.train(
        max_epochs=50,  # Granular might need slightly longer to converge
        batch_size=2048,
        accelerator="mps",
        devices=1,
        early_stopping=True,
        early_stopping_patience=4,
        early_stopping_min_delta=0.001,
        plan_kwargs={"lr": 5e-4, "weight_decay": 1e-4, "reduce_lr_on_plateau": True},
        logger=logger
    )

    print("✅ Training Complete. Moving to Dual-Head Wrapper...")

    # ==========================================
    # IN-MEMORY GUT CHECK (Pre-Save)
    # ==========================================
    print("\n" + "=" * 40)
    print(f"🧠 LIVE IN-MEMORY GUT CHECK ({adata_tensor.var_names.shape} GENES)")
    print("=" * 40)

    out_dir = f"../H&E Computer Vision/RNA_Cell_Classification/models/v{version}"

    # ==========================================
    # 1. SAVE NATIVE SCVI MODELS (For Python Backups)
    # ==========================================
    broad_scanvi_model.save(f"{out_dir}/broad_scvi_model", overwrite=True)
    granular_scanvi_model.save(f"{out_dir}/granular_scvi_model", overwrite=True)
    print("✅ Native scVI models saved to disk.")

    # ==========================================
    # 1.5 THE PREDICTION AUDIT
    # ==========================================
    print("\n--- Running Prediction Audit ---")

    # Broad Audit
    print("Analyzing Broad Accuracy...")
    broad_val_preds = broad_scanvi_model.predict(adata_tensor)
    broad_report = classification_report(
        adata_tensor.obs['mechanical_cell_type'],
        broad_val_preds,
        output_dict=True,
        zero_division=0
    )

    # Granular Audit
    print("Analyzing Granular Accuracy...")
    granular_val_preds = granular_scanvi_model.predict(adata_tensor)
    granular_report = classification_report(
        adata_tensor.obs['granular_cell_type'],
        granular_val_preds,
        output_dict=True,
        zero_division=0
    )

    # Save reports to review later
    with open(f"{out_dir}/audit_broad.json", "w") as f:
        json.dump(broad_report, f, indent=4)
    with open(f"{out_dir}/audit_granular.json", "w") as f:
        json.dump(granular_report, f, indent=4)

    # The baseline transcription level expected by the Rust physics engine.
    depth_data = {
        "target_depth": final_mean_depth,
        "total_cells_trained": adata_tensor.n_obs
    }

    with open(f"{out_dir}/target_depth.json", "w") as f:
        json.dump(depth_data, f, indent=4)

    print(f"📦 Exported target_depth.json ({final_mean_depth:.2f})")

    # Print a quick warning for difficult granular classes
    print("\n⚠️ Granular Sub-Types Requiring LR-Voting (F1 < 0.70):")
    for cell_type, metrics in granular_report.items():
        if isinstance(metrics, dict) and metrics.get('f1-score', 1.0) < 0.70:
            print(f"   - {cell_type}: {metrics['f1-score']:.2f}")

    # ==========================================
    # 2. EXTRACT CLASS MAPPINGS
    # ==========================================
    broad_registry = broad_scanvi_model.adata_manager.get_state_registry("labels")
    broad_classes = broad_registry.categorical_mapping.tolist()

    granular_registry = granular_scanvi_model.adata_manager.get_state_registry("labels")
    granular_classes = granular_registry.categorical_mapping.tolist()

    print(f"✅ Verified Broad labels ({len(broad_classes)}): {broad_classes}")
    print(f"✅ Verified Granular labels ({len(granular_classes)}): {granular_classes[:5]}...")

    # Save both to JSON
    with open(f"{out_dir}/broad_class_map.json", "w") as f:
        json.dump(broad_classes, f)
    with open(f"{out_dir}/granular_class_map.json", "w") as f:
        json.dump(granular_classes, f)

    # ==========================================
    # 3. EXPORT DATA ARTIFACTS (Genes & Means)
    # ==========================================
    model_genes = adata_tensor.var_names.tolist()
    with open(f"{out_dir}/model_genes.json", "w") as f:
        json.dump(model_genes, f)

    if type(adata_tensor.X).__name__ in ["csr_matrix", "csc_matrix"]:
        model_means = adata_tensor.X.mean(axis=0).A1.tolist()
    else:
        model_means = adata_tensor.X.mean(axis=0).tolist()

    with open(f"{out_dir}/model_means.json", "w") as f:
        json.dump(model_means, f)
    print(f"📦 Exported Genes ({len(model_genes)}) & Means")

    # ==========================================
    # 4. FUSE THE DUAL-HEAD WRAPPER
    # ==========================================
    print("\n--- Assembling Dual-Head Inference Model ---")
    number_of_genes = len(model_genes)

    # 1. Extract ONLY the state dicts and immediately move them to CPU
    # This breaks the link to the heavy scANVI objects
    broad_sd_raw = {k: v.cpu().clone() for k, v in broad_scanvi_model.module.state_dict().items()}
    granular_sd_raw = {k: v.cpu().clone() for k, v in granular_scanvi_model.module.state_dict().items()}

    # 2. NUKE the massive training models to free up GBs of RAM
    del broad_scanvi_model
    del granular_scanvi_model
    del pan_scvi_model
    gc.collect()
    print("🧹 Training models purged from RAM to prevent Segfault.")

    # 3. Perform the fusion on the raw CPU state dicts
    # We update the fusion function to take dicts, not the full models
    num_broad = len(broad_classes)
    num_granular = len(granular_classes)

    # def extract_fused_weights_safe(b_sd, g_sd, nb, ng):
    #     new_sd = {}
    #
    #     # Map Z-Encoder (from granular)
    #     # Use .clone() to ensure we aren't pointing to deleted memory
    #     new_sd['z_encoder_l0.weight'] = g_sd['z_encoder.encoder.fc_layers.Layer 0.0.weight'].clone()
    #     new_sd['z_encoder_l0.bias'] = g_sd['z_encoder.encoder.fc_layers.Layer 0.0.bias'].clone()
    #     # ... (Repeat for all z_encoder and BN keys using .clone()) ...
    #
    #     # Map Broad with Padding
    #     b_w = b_sd['classifier.classifier.1.weight'].clone()
    #     b_b = b_sd['classifier.classifier.1.bias'].clone()
    #     if b_w.shape[0] < nb:
    #         b_w = torch.cat([b_w, torch.zeros(nb - b_w.shape[0], b_w.shape[1])], dim=0)
    #         b_b = torch.cat([b_b, torch.zeros(nb - b_b.shape[0])], dim=0)
    #     new_sd['broad_out.weight'] = b_w
    #     new_sd['broad_out.bias'] = b_b
    #
    #     # ... (Repeat for granular_out with .clone() and padding) ...
    #
    #     return new_sd

    # 4. Initialize and Load
    clean_dual_model = DualHeadSCANVI(input_genes=473, num_broad_classes=num_broad, num_granular_classes=num_granular)
    fused_sd = extract_dual_head_weights(broad_sd_raw, granular_sd_raw, num_broad, num_granular)

    clean_dual_model.load_state_dict(fused_sd, strict=True)  # Use False temporarily to debug
    print("✅ Weights bridged on CPU.")
    clean_dual_model.eval()
    print("✅ Dual-Head Weights Fused Successfully.")

    # ==========================================
    # 5. SAFETENSORS EXPORT (Clean Architecture)
    # ==========================================
    # We export the CLEAN model, not the messy scVI module
    clean_sd_for_export = {
        key: tensor.cpu().contiguous()
        for key, tensor in clean_dual_model.state_dict().items()
    }

    save_file(clean_sd_for_export, f"{out_dir}/dual_head_spatial.safetensors")
    print("📦 Exported Safetensors (Clean Architecture)")

    # ==========================================
    # 6. ONNX EXPORT (Dynamic Shapes + Dual Outputs)
    # ==========================================
    dummy_input = torch.randn(1, number_of_genes, device="cpu")
    batch = torch.export.Dim("batch", min=1, max=8192)
    dynamic_shapes = {"x": {0: batch}}

    print("📦 Exporting ONNX with Opset 18...")
    torch.onnx.export(
        clean_dual_model,  # <-- FIX: Exporting the clean dual wrapper, NOT pan_scanvi_model!
        dummy_input,
        f"{out_dir}/dual_head_classifier.onnx",
        export_params=True,
        opset_version=18,
        input_names=['x'],
        output_names=['latent', 'logits_broad', 'logits_granular'],  # <-- NEW: 3 Outputs!
        dynamic_shapes=dynamic_shapes
    )
    print("✅ ONNX export successful.")

    print("\n--- Packing ONNX into a single file ---")
    onnx_path = f"{out_dir}/dual_head_classifier.onnx"
    combined_path = f"{out_dir}/dual_head_classifier_single.onnx"

    import onnx
    # 1. Load the model (This automatically pulls the .data file into memory)
    model = onnx.load(onnx_path)

    # 2. Save it back out, explicitly forcing all data into the main protobuf file
    onnx.save_model(model, combined_path, save_as_external_data=False)

    # 3. Clean up the old split files to avoid confusion
    if os.path.exists(f"{onnx_path}.data"):
        os.remove(f"{onnx_path}.data")
    os.remove(onnx_path)

    # Rename the combined file back to the original name
    os.rename(combined_path, onnx_path)
    print("✅ ONNX packed into a single self-contained file.")

    print("\n🚀 Full Pipeline Complete.")


if __name__ == "__main__":
    main()