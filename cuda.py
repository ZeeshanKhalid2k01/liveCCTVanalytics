import torch

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

if cuda_available:
    # Check CUDA version
    cuda_version = torch.version.cuda
    print("CUDA Version:", cuda_version)

    # Check cuDNN version
    cudnn_version = torch.backends.cudnn.version()
    print("cuDNN Version:", cudnn_version)

# Check PyTorch version
pytorch_version = torch.__version__
print("PyTorch Version:", pytorch_version)

import torch
print(torch.cuda.is_available())
# True


import torch

# Check PyTorch version
print("PyTorch Version:", torch.__version__)

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())

# If CUDA is available, print CUDA version and device configuration
if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("Device Count:", torch.cuda.device_count())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Device Capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))