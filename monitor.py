import pynvml


class GPUInfo:

    def __init__(self, gpu_handel):
        self._handle = gpu_handel


class GPUMonitor:

    def __init__(self):
        self.server_info_list = []
        self.UNIT = 1024 * 1024
        self._monitor = pynvml.nvmlInit()  # 初始化
        self.gpu_device_count = self._monitor.nvmlDeviceGetCount()

    def __init(self):
        for gpu_idx in range(self.gpu_device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(2)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)



