import functools
import gc
import inspect
import psutil
import torch
from functools import partial


def should_reduce_batch_size(exception: Exception) -> bool:
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


def find_executable_batch_size(function, starting_batch_size=128):
    batch_size = starting_batch_size
    gc.collect()
    torch.cuda.empty_cache()
    # Guard against user error
    while True:
        if batch_size == 0:
            raise RuntimeError("No executable batch size found, reached zero.")
        try:
            res = function(batch_size=batch_size)
            # print(batch_size, res)
            return batch_size
        except Exception as e:
            if should_reduce_batch_size(e):
                gc.collect()
                torch.cuda.empty_cache()
                batch_size //= 2
            else:
                raise


def get_gpu_info():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return [{}]

    gpu_count = torch.cuda.device_count()
    gpu_info = []

    for i in range(gpu_count):
        device_name = torch.cuda.get_device_name(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 2  # Convert bytes to MB
        # Allocate a small tensor to force PyTorch to initialize the device and free up memory
        torch.cuda.empty_cache()
        torch.cuda.set_device(i)
        free_memory = torch.cuda.memory_reserved(i) / 1024 ** 2  # Convert bytes to MB
        used_memory = torch.cuda.memory_allocated(i) / 1024 ** 2  # Convert bytes to MB
        available_memory = total_memory - used_memory

        info = {
            'id': i,
            'name': device_name,
            'total_memory': total_memory,
            'free_memory': available_memory,
            'used_memory': used_memory
        }
        gpu_info.append(info)
    gpu_info = sorted(gpu_info, key=lambda g: g['free_memory'], reverse=True)
    return gpu_info


if __name__ == '__main__':
    import torch

    res = get_gpu_info()
    print(res)
