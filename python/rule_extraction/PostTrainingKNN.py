from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# --- PHASE 1: Build the Chemical Dictionary ---
# We train a lightweight KNN classifier on the Latent Space of your Reference Atlas
print("Training the Granular Label transfer model...")

# Extract the latent coordinates and the granular labels from your training data
X_train_latent = adata_pan_raw.obsm["X_scANVI"]
y_train_granular = adata_pan_raw.obs['granular_state'] # The column you safely saved!

# Initialize the KNN (15 neighbors is the gold standard for single-cell)
# 'distance' weighting means closer neighbors have a stronger vote
knn_chemical = KNeighborsClassifier(n_neighbors=15, weights='distance')
knn_chemical.fit(X_train_latent, y_train_granular)

print("✅ Chemical Dictionary Trained.")


# --- PHASE 2: Inference on New Spatial Data ---
# Imagine 'spatial_adata' is a new Xenium/CosMx slide you just ran through your pipeline
# 1. First, scANVI predicts the Broad Mechanical Class
spatial_adata.obs['mechanical_class'] = model.predict(spatial_adata)

# 2. Extract the Latent Space coordinates for the new spatial cells
X_new_latent = spatial_adata.obsm["X_scANVI"]

# 3. Use the KNN to predict the Granular State for the interactome
spatial_adata.obs['chemical_class'] = knn_chemical.predict(X_new_latent)

print("Mapping Complete. Sample output:")
print(spatial_adata.obs[['mechanical_class', 'chemical_class']].head())