import onnx

# 1. Load the 2KB skeleton (it automatically looks for the .data file in the same folder)
model = onnx.load("../H&E Computer Vision/RNA_Cell_Classification/models/Dualv1/dual_head_classifier.onnx")

# 2. Save it as a single unified file.
# 'save_as_external_data=False' is the key: it forces everything into one file.
onnx.save_model(
    model,
    "../H&E Computer Vision/RNA_Cell_Classification/models/Dualv1/rna_cell_classifier.onnx",
    save_as_external_data=False
)

print("✅ Model unified! You can now delete the .data file.")