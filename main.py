
if __name__ == "__main__":
    print("Starting SKOUT Engine...")
    # client = SynergyClient()

# Sanity Check
def assert_cuda_ready():
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.cuda.synchronize()

