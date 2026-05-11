# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from monai.networks.nets import UNet

# 1. Minimal setup
device = torch.device("cpu")
model = UNet(
    spatial_dims=3, in_channels=4, out_channels=1,
    channels=(16, 32), strides=(2,), num_res_units=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_function = torch.nn.MSELoss()

# 2. Dummy data (Simulating 1 patch: 4 channels, 32x32x32)
dummy_input = torch.randn(1, 4, 32, 32, 32)
dummy_target = torch.randn(1, 1, 32, 32, 32)

# 3. The Test Loop
print("Starting Sanity Check...")
for i in range(5):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = loss_function(output, dummy_target)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}/5 - Loss: {loss.item():.6f}")