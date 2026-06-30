import onnxruntime as ort
import numpy as np
import json


def test_onnx_model():
    print("Loading model and mappings...")

    # 1. Load the class mapping dictionary
    with open("class_mapping.json", "r") as f:
        class_mapping = json.load(f)

    # 2. Load the ONNX model into the runtime engine
    session = ort.InferenceSession("universal_cell_classifier.onnx")

    # 3. Create a dummy "cell" (an array of 2000 floats, representing gene counts)
    # We use random noise here, but in production, this is your Xenium data.
    dummy_genes = np.random.rand(1, 2000).astype(np.float32)

    # 4. Run Inference!
    inputs = {session.get_inputs()[0].name: dummy_genes}
    logits = session.run(None, inputs)[0]

    # 5. Apply Softmax to get percentages
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # 6. Find the winning class and its confidence
    predicted_index = np.argmax(probabilities[0])
    confidence = probabilities[0][predicted_index] * 100

    predicted_cell_type = class_mapping[str(predicted_index)]

    print("\n" + "=" * 40)
    print("🔬 INFERENCE TEST SUCCESSFUL")
    print("=" * 40)
    print(f"Predicted Class ID: {predicted_index}")
    print(f"Biological Name:    {predicted_cell_type}")
    print(f"Confidence:         {confidence:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    test_onnx_model()