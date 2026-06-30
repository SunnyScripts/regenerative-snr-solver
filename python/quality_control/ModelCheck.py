import torch

# Path to the scvi model file
model_path = "../Cell Classifier/v14/model.pt"

# weights_only=False is required for scvi models because they contain
# custom metadata (pickled objects) beyond just weight tensors.
try:
    data = torch.load(model_path, map_location="cpu", weights_only=False)

    # scvi-tools stores the class name in the registry within the 'attr_dict'
    registry = data.get('attr_dict', {}).get('registry_', {})
    model_class = registry.get('model_class_name')

    print("\n--- MODEL IDENTITY CHECK ---")
    if model_class:
        print(f"✅ Success! This model is a: {model_class}")
    else:
        print("❌ Model loaded, but no class name found in registry.")

    # Let's also check the labels mapping while we are here
    labels_state = data.get('attr_dict', {}).get('manager_state_registry', {}).get('labels', {})
    mapping = labels_state.get('categorical_mapping')
    if mapping is not None:
        print(f"✅ Categorical Mapping: {mapping}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")