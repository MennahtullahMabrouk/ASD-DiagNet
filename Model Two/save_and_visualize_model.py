import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# Define the MTAutoEncoder model (same as in your original code)
class MTAutoEncoder(nn.Module):
    def __init__(self, num_inputs, num_latent, tied=True, use_dropout=False):
        super(MTAutoEncoder, self).__init__()
        self.tied = tied
        self.num_latent = num_latent
        self.fc_encoder = nn.Linear(num_inputs, num_latent)
        if not tied:
            self.fc_decoder = nn.Linear(num_latent, num_inputs)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5) if use_dropout else nn.Identity(),
            nn.Linear(num_latent, 1)
        )

    def forward(self, x, eval_classifier=False):
        x = torch.tanh(self.fc_encoder(x))
        x_logit = self.classifier(x) if eval_classifier else None
        x_rec = F.linear(x, self.fc_encoder.weight.t()) if self.tied else self.fc_decoder(x)
        return x_rec, x_logit

# Initialize the model
num_inputs = 100  # Example input size (adjust based on your data)
num_latent = 50   # Example latent size (adjust based on your data)
model = MTAutoEncoder(num_inputs=num_inputs, num_latent=num_latent, tied=True, use_dropout=False)

# Save the model to a file
torch.save(model.state_dict(), 'asdmodeltwo.sh')
print("Model saved as 'asdmodeltwo.sh'.")

# Generate the model architecture visualization
# Create a dummy input tensor
dummy_input = torch.randn(1, num_inputs)  # Batch size of 1, input size of num_inputs

# Forward pass to get the output
rec, logits = model(dummy_input, eval_classifier=True)

# Visualize the model architecture
dot = make_dot(rec, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('asdmodeltwo_architecture')  # Saves the architecture as 'asdmodeltwo_architecture.png'
print("Model architecture saved as 'asdmodeltwo_architecture.png'.")