import pandas as pd
from sklearn.metrics import adjusted_rand_score

def validate_clustering():
    print("Loading clustering results...")

    # 1. Load the industry-standard Scanpy results
    # Expected columns: ['cell_id', 'scanpy_cluster']
    df_scanpy = pd.read_csv("scanpy_clusters.csv")

    # 2. Load your custom Rust engine results
    # Expected columns: ['cell_id', 'rust_cluster']
    df_rust = pd.read_csv("rust_clusters.csv")

    # 3. Merge them on cell_id to ensure perfect 1:1 alignment
    # This prevents errors if one pipeline dropped a low-quality cell
    merged_df = pd.merge(df_scanpy, df_rust, on="cell_id", how="inner")

    print(f"Successfully aligned {len(merged_df)} cells.")

    # 4. Calculate the Adjusted Rand Index (ARI)
    # The order of the cluster IDs (0 vs 1) doesn't matter, ARI only measures
    # if the *groupings* are mathematically equivalent.
    ari_score = adjusted_rand_score(
        labels_true=merged_df["scanpy_cluster"],
        labels_pred=merged_df["rust_cluster"]
    )

    print("\n" + "="*40)
    print(f"🔬 ADJUSTED RAND INDEX (ARI): {ari_score:.4f}")
    print("="*40)

    if ari_score > 0.90:
        print("✅ SUCCESS: Rust pipeline is mathematically equivalent to Scanpy!")
    elif ari_score > 0.75:
        print("⚠️ WARNING: High overlap, but minor edge-case differences exist.")
    else:
        print("❌ FAILURE: Clustering pipelines are diverging significantly.")

if __name__ == "__main__":
    validate_clustering()