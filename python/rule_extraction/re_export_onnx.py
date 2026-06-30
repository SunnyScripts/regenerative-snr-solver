import torch
import torch.nn as nn
from safetensors.torch import save_file

class UniversalCellClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

# 1. Load the model natively
model = UniversalCellClassifier(input_dim=2000, num_classes=848)
model.load_state_dict(torch.load("universal_cell_classifier_backup.pt", weights_only=True))

# 2. Extract the raw dictionaries of just the weights and biases
state_dict = model.state_dict()

# 3. Save as a pure binary format (No graphs, no ONNX ops, just raw numbers)
save_file(state_dict, "classifier_weights.safetensors")
print("✅ Weights safely extracted to classifier_weights.safetensors")