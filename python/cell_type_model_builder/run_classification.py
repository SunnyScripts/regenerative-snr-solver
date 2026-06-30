import pandas
import scanpy
import scvi
from .utilities import *

from pathlib import Path
LOCAL_DIR = Path(__file__).resolve().parent

class RNACellClassifier:
    def __init__(self, model_weights_path: str):
        self.model_weights = model_weights_path

    def generate_labeled_parquet(self,
         cells_parquet: str,
         features_h5: str,
         model_version: int,
         output_parquet: str
     ) -> str:
        """
        Loads the Teacher model, runs inference on the RNA data, and outputs
        a new parquet containing ['x', 'y', 'cell_id', 'predicted_label'].
        """

        print("1. Loading physical coordinates...")
        cells_df = pandas.read_parquet(cells_parquet)
        cells_df['cell_id'] = cells_df['cell_id'].astype(str)

        print("2. Loading 10x Xenium H5 matrix...")
        adata = scanpy.read_10x_h5(features_h5)

        print("3. Aligning spatial data to Teacher genes...")
        spatial_tensor, final_gene_order = align_spatial_to_teacher_offline(
            adata,
            LOCAL_DIR / "model_genes.json",
            LOCAL_DIR / "symbol2entrez.json"
        )

        print("4. Creating Dummy AnnData for scVI Native Inference...")
        # Wrap the tensor back into an AnnData object so scvi can read it natively
        adata_spatial = scanpy.AnnData(spatial_tensor.cpu().numpy())
        adata_spatial.var_names = final_gene_order
        adata_spatial.obs_names = adata.obs_names

        # CRITICAL: Satisfy scVI's internal setup registry
        # These columns must exist to load the model, even if the data is just a placeholder
        adata_spatial.obs['batch_tissue'] = "Skin"
        adata_spatial.obs['mechanical_cell_type'] = "Unknown"

        print("5. Loading Native Teacher Model...")
        # This bypasses the safetensors mismatch. It loads the exact architecture and weights perfectly.
        model = scvi.model.SCANVI.load(f"{LOCAL_DIR}/models/v{model_version}", adata=adata_spatial)

        print("6. Running Native Inference...")
        # Get the actual string predictions directly (No JSON mapping needed!)
        predicted_labels = model.predict(adata_spatial)

        # Get the confidences
        soft_probs = model.predict(adata_spatial, soft=True)
        confidences = soft_probs.max(axis=1).values

        # 7. Analyze and Output
        results_df = pandas.DataFrame({
            'category': predicted_labels,
            'confidence': confidences
        })

        print("\n--- CONFIDENCE REPORT ---")
        print(results_df.groupby('category')['confidence'].mean())

        # (Merge with cells_df and save to parquet here)
        print("5. Merging Biology with Geometry...")
        # 5. Mapping and Merging
        # class_mapping = {0: "Endothelial", 1: "Epithelial", 2: "Immune/Fluid", 3: "Mesenchymal"}
        # predicted_labels = [class_mapping[idx] for idx in predicted_labels]

        labels_df = pandas.DataFrame({
            'cell_id': adata.obs.index.astype(str),
            'broad_category': predicted_labels
        })

        labeled_cells = pandas.merge(cells_df, labels_df, on='cell_id', how='inner')

        # 6. Biological Sanity Check: Ratio Report
        print("\n" + "=" * 30)
        print("TEACHER MODEL RATIO REPORT")
        print("=" * 30)

        counts = labeled_cells['broad_category'].value_counts()
        percentages = labeled_cells['broad_category'].value_counts(normalize=True) * 100

        report = pandas.DataFrame({
            'Count': counts,
            'Ratio (%)': percentages
        })

        print(report.to_string(formatters={'Ratio (%)': '{:,.2f}%'.format}))
        print("=" * 30 + "\n")

        # 7. Safety check for "Zero-Class" failure
        # if len(counts) < 5:
        #     missing = set(class_mapping.values()) - set(counts.index)
        #     print(f"⚠️ WARNING: Teacher failed to predict any: {missing}")
        #     print("Check gene index alignment or softmax temperature in training.")

        # 8. Add Bounding Boxes and Save
        window_radius = 64
        labeled_cells['bbox_xmin'] = labeled_cells['x_centroid'] - window_radius
        labeled_cells['bbox_xmax'] = labeled_cells['x_centroid'] + window_radius
        labeled_cells['bbox_ymin'] = labeled_cells['y_centroid'] - window_radius
        labeled_cells['bbox_ymax'] = labeled_cells['y_centroid'] + window_radius

        labeled_cells.to_parquet(output_parquet)
        print("Labeled cells saved")

        return output_parquet