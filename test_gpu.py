import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU count: {torch.cuda.device_count()}")

# Try to create a tensor on GPU
try:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print("\n✓ Successfully performed matrix multiplication on GPU!")
    print(f"Result shape: {z.shape}")
    print(f"Result device: {z.device}")
except Exception as e:
    print(f"\n✗ Error using GPU: {e}")
