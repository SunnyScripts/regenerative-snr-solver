import json

import pandas
import torch
from torch import nn
from safetensors.torch import load_file, load_model


def align_spatial_to_teacher_offline(
        adata_spatial,
        entrez_list_path="back_entrez_list.json",
        symbol_map_path="back_spatial_map.json"
):
    print(f"1. Initial spatial shape: {adata_spatial.shape}")

    # 1. Load the static, deterministic JSON files
    print("2. Loading training gene mappings...")
    with open(entrez_list_path, 'r') as f:
        teacher_entrez_list = json.load(f)

    with open(symbol_map_path, 'r') as f:
        symbol_to_entrez = json.load(f)

    print(f"   -> Teacher strictly expects {len(teacher_entrez_list)} genes.")

    # 2. Translate Spatial Symbols to Entrez IDs using the static dictionary
    print("3. Translating Spatial Symbols...")
    # If a symbol isn't in the JSON map, we leave it as a symbol.
    # It will automatically be deleted in the next step anyway.
    adata_spatial.var_names = [symbol_to_entrez.get(sym, sym) for sym in adata_spatial.var_names]
    adata_spatial.var_names_make_unique()

    # 3. Force Strict Column Alignment
    print("4. Forcing Strict Tensor Alignment...")
    # Convert to dense DataFrame for easy column manipulation
    spatial_df = pandas.DataFrame(
        adata_spatial.X.toarray() if hasattr(adata_spatial.X, "toarray") else adata_spatial.X,
        index=adata_spatial.obs.index,
        columns=adata_spatial.var_names
    )

    # The Magic Trick: Reindex drops extra genes, orders the remaining ones perfectly,
    # and fills in any missing genes with 0.0 expression instantly.
    aligned_df = spatial_df.reindex(columns=teacher_entrez_list, fill_value=0.0)

    print(f"5. Final aligned shape: {aligned_df.shape} (Should be N x {len(teacher_entrez_list)})")

    # 4. Convert back to PyTorch Tensor (ready for Apple Silicon)
    aligned_tensor = torch.tensor(aligned_df.values, dtype=torch.float32)

    return aligned_tensor, aligned_df.columns.tolist()

class DynamicSCANVIClassifier(nn.Module):
    def __init__(self, config_path, input_genes=449, num_classes=4):
        super().__init__()

        # Load the config dynamically
        with open(config_path, 'r') as f:
            config = json.load(f)['non_kwargs']

        n_hidden = config['n_hidden']  # 128
        n_latent = config['n_latent']  # 30
        n_layers = config['n_layers']  # 2
        dropout_rate = config['dropout_rate']  # 0.1

        # Build the Encoder (Genes -> Hidden -> Latent)
        layers = []
        in_dims = input_genes
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dims, n_hidden))
            layers.append(nn.BatchNorm1d(n_hidden))  # scvi uses BatchNorm by default
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dims = n_hidden

        self.encoder = nn.Sequential(*layers)

        # Latent Bottleneck
        self.latent = nn.Linear(n_hidden, n_latent)

        # Classifier (Latent -> Broad Categories)
        self.classifier = nn.Linear(n_latent, num_classes)

    def forward(self, x):
        # Pass input through the architecture
        x = self.encoder(x)
        latent_space = self.latent(x)           # Save this state!
        logits = self.classifier(latent_space)  # Pass it to the classifier
        return latent_space, logits             # Return BOTH


def test_safetensors_load(config_path="model_config.json", safetensors_path="model.safetensors"):
    print("1. Building Dynamic Architecture from JSON...")
    model = DynamicSCANVIClassifier(config_path=config_path, input_genes=449, num_classes=4)

    print("2. Loading Safetensors...")
    # Read the raw dictionary keys to check for scvi's internal naming
    weights = load_file(safetensors_path)
    print("\nSample of saved weight keys:")
    for key in list(weights.keys())[:5]:
        print(f" - {key}")

    try:
        # Attempt to pour the weights in.
        # strict=False ignores the decoder/dispersion weights we don't need for inference
        load_model(model, safetensors_path, strict=False)
        print("\n✅ Success: Safetensors loaded into pure PyTorch architecture!")
    except Exception as e:
        print(f"\n❌ Key Mismatch Error: {e}")
        print("Note: scvi uses specific prefix names (like 'module.z_encoder...'). We may need to strip prefixes.")