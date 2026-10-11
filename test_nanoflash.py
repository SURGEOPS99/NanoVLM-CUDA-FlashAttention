"""Standalone GPU correctness tests for custom NanoFlash CUDA forward/backward."""
import torch
from nanoflash_cuda import numerical_self_test, register_nanoflash

print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
assert torch.cuda.is_available(), 'Run from CUDA-enabled WSL virtual environment'
print('GPU:', torch.cuda.get_device_name(0))
register_nanoflash()
assert numerical_self_test()
print('PASS: All NanoFlash CUDA parity cases')
