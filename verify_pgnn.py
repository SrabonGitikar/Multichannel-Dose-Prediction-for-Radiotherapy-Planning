import torch
from train_monai import ClinicalDoseLoss

def test_pgnn_loss():
    print("Initializing PGNN Loss Function...")
    loss_fn = ClinicalDoseLoss(d_prescription=60.0, max_bladder=40.0, max_rectum=45.0)
    
    # Batch=2, Channels=4, D=32, H=32, W=32
    print("Creating dummy tensors (Batch=2, Channels=4, D=32, H=32, W=32)...")
    inputs = torch.randn(2, 4, 32, 32, 32, requires_grad=False)
    
    # Simulate binary mask for PTV (Ch 1)
    inputs[:, 1:2] = (torch.rand(2, 1, 32, 32, 32) > 0.8).float()
    
    # Simulate SDM for Bladder (Ch 2) and Rectum (Ch 3) (Negative is inside)
    inputs[:, 2:3] = torch.randn(2, 1, 32, 32, 32) * 50.0
    inputs[:, 3:4] = torch.randn(2, 1, 32, 32, 32) * 50.0
    
    pred_dose = torch.randn(2, 1, 32, 32, 32) * 50.0 + 30.0 # Mean dose 30Gy
    pred_dose.requires_grad = True
    true_dose = torch.randn(2, 1, 32, 32, 32) * 50.0 + 30.0
    
    print("Calculating Loss...")
    loss = loss_fn(pred_dose, true_dose, inputs)
    
    print(f"Total Loss Calculated: {loss.item():.4f}")
    
    print("Testing backward pass (Gradient calculation)...")
    loss.backward()
    
    if pred_dose.grad is not None:
        print("SUCCESS: Gradients successfully propagated back to predicted dose tensor.")
        print(f"Gradient Tensor Shape: {pred_dose.grad.shape}")
        print(f"Max Gradient: {pred_dose.grad.abs().max().item():.4f}")
    else:
        print("FAILED: No gradients found.")

if __name__ == "__main__":
    test_pgnn_loss()
