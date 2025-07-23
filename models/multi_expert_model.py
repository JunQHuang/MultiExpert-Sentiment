from torch import nn
import json
import torch
import gc
from utils.gpu_utils import get_gpu_info, find_executable_batch_size
from functools import partial
import queue
import concurrent
import time
import threading


unit = 1024 * 1024


class GPUMonitor:

    def __init__(self, period=1):
        self.gpu_utility = 0
        self.gpu_time = 0
        self.period = period
        self.run = False

    def start(self):
        total_memory_used = 0
        total_memory = 0
        count = 0
        self.run = True
        while self.run is True:
            gpu_info_list = get_gpu_info()
            for gpu_info in gpu_info_list:
                total_memory_used += gpu_info['used_memory']
                total_memory += gpu_info['total_memory']
            count += 1
            time.sleep(self.period)
        self.gpu_time = count
        self.gpu_utility = total_memory_used / total_memory

    def stop(self):
        self.run = False

    def show(self):
        return self.gpu_utility, self.gpu_time


class MultiExpertMonitor(nn.Module):

    def __init__(self, init_functions, forward_functions, forward_kwargs, input_data,
                 gate_method, gpu_usage_config=None,
                 estimate_batch_size=256, max_threading=4):
        super(MultiExpertMonitor, self).__init__()
        self.max_threading = max_threading
        self.usable_devices = get_gpu_info()
        self._init_models = init_functions
        self.forwards = forward_functions
        self.forward_kwargs = forward_kwargs
        self.gate_method = gate_method
        assert len(self._init_models) == len(forward_functions) == len(forward_kwargs)
        self._init(gpu_usage_config, estimate_batch_size, input_data)
        self._init_now = False
        self.infer_batch_size = estimate_batch_size

    def _init(self, gpu_usage_config, estimate_batch_size, input_data):
        self.gpu_usage_config = self.analysis_input_models(input_data, gpu_usage_config, estimate_batch_size)
        # self.

    def analysis_input_models(self, input_data, gpu_usage_config=None, estimate_batch_size=256):
        # print(self.usable_devices[0])
        import pynvml
        import time
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        device = "cuda:{}".format(self.usable_devices[0]["id"])
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory / unit
        self.infer_batch_size = estimate_batch_size
        if gpu_usage_config is None:
            gpu_usage_config = {}
            print("开始逐一初始化模型并测试缓存消耗, 会花一些时间")
            i = 0
            for _init_model, _forward_fn, _init_kwargs in zip(self._init_models, self.forwards, self.forward_kwargs):
                # print("before ", torch.cuda.memory_allocated() / unit)
                gpu_before = torch.cuda.memory_allocated() / unit
                model = _init_model(device)
                # print("init ", torch.cuda.memory_allocated() / unit)
                gpu_load = torch.cuda.memory_allocated() / unit
                with torch.no_grad():
                    test_fn = partial(_forward_fn, model=model, input_data=input_data, **_init_kwargs)
                    infer_batch_size = find_executable_batch_size(test_fn, estimate_batch_size)
                    if infer_batch_size < self.infer_batch_size:
                        self.infer_batch_size = infer_batch_size
                gpu_infer = torch.cuda.memory_allocated() / unit
                # print("infer ", torch.cuda.memory_allocated() / unit)
                del model
                del test_fn
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # time.sleep(10)
                gpu_after = torch.cuda.memory_allocated() / unit
                max_gpu_use = torch.cuda.max_memory_reserved() / unit
                torch.cuda.reset_peak_memory_stats(0)
                # print(gpu_before, gpu_load, gpu_infer, gpu_after, max_gpu_use)
                model_memory_usage = gpu_load - gpu_before
                infer_memory_usage = max_gpu_use - gpu_before
                # average_gpu_usage = (gpu_load + gpu_infer + gpu_after) / 3
                average_usage_rate = (infer_memory_usage / total_gpu_memory) * 100
                gpu_usage_config[i] = {
                    "model": model_memory_usage,
                    "infer": infer_memory_usage,
                    "average_usage_rate": average_usage_rate
                }
                i += 1
            return gpu_usage_config
        else:
            return gpu_usage_config

    def get_cur_device(self, gpu_config):
        model_gpu_usage = gpu_config["infer"]
        gpu_info = get_gpu_info()
        cur_max_memory = gpu_info[0]["free_memory"]
        print("cur_max_memory, model_gpu_usage: ",cur_max_memory, model_gpu_usage)
        while model_gpu_usage > cur_max_memory:
            print("没有可用的gpu, 等待一下")
            time.sleep(10)
            gpu_info = get_gpu_info()
            cur_max_memory = gpu_info[0]["free_memory"]
        device = "cuda:{}".format(gpu_info[0]["id"])
        return device

    def _forward(self, texts, task):
        init_model, forward_fn, kwargs = task["fns"]
        print("task starts: ", init_model)
        gpu_config = task["config"]

        while self._init_now is True:
            time.sleep(10)
        device = None
        while device is None:
            device = self.get_cur_device(gpu_config)
        print("init start: ", init_model)
        self._init_now = True
        model = init_model(device)
        time.sleep(1)
        print("init finished: ", init_model)
        self._init_now = False

        pred = forward_fn(model, texts, batch_size=self.infer_batch_size, **kwargs)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print("task finished: ", init_model)
        return pred

    def merge(self, results):
        res_1, res_2, res_3, res_4 = results
        # res_2,res_3,res_4 = results
        final_res = self.gate_method(res_1, res_2, res_3, res_4)
        # final_res = self.gate_method(res_2,res_3,res_4)
        return final_res

        
    def model_forward(self, input_data):
        gc.collect()
        torch.cuda.empty_cache()
        start_time = time.time()  # 记录推断开始时间
        with torch.no_grad():
            task_list = []
            model_idx = 0
            for _init_model, _forward_fn, _init_kwargs in zip(self._init_models, self.forwards, self.forward_kwargs):
                model_idx == 2
                _gpu_config = self.gpu_usage_config[model_idx]
                task = {"fns": [_init_model, _forward_fn, _init_kwargs], "config": _gpu_config}
                task_list.append(task)
                model_idx += 1
            sorted_task_list = sorted(task_list, key=lambda t: t['config']["infer"], reverse=True)
            tasks = queue.Queue()
            for task in sorted_task_list:
                tasks.put(task)

            # monitor_stop_event = concurrent.futures.thread.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threading) as executor:
                futures = []
                # monitor_future = executor.submit(monitor_memory, monitor_stop_event, 0.5)
                while not tasks.empty():
                    task = tasks.get()
                    futures.append(executor.submit(self._forward, input_data, task))
                concurrent.futures.wait(futures)
            results = []
            for future in futures:
                results.append(future.result())
            end_time = time.time()  # 记录推断结束时间
            inference_time = end_time - start_time  # 计算推断总时长
            final_result = self.merge(results)
        return final_result, inference_time  # 返回结果和推断时间

    def forward(self, input_data):
        monitor = GPUMonitor()
        monitor_thread = threading.Thread(target=monitor.start)
        try:
            print("model forward starts!!!")
            monitor_thread.start()
            res, inference_time = self.model_forward(input_data)  # 获取结果和推断时间
        finally:
            monitor.stop()
            monitor_thread.join()
        gpu_usage_info = monitor.show()
        self.print_gpu_usage_config()  # 输出 GPU 使用信息
        print(f"Inference Time: {inference_time:.2f} seconds")  # 打印推断时间
        return res, gpu_usage_info, inference_time  # 返回结果、GPU 使用信息和推断时间

    #输出 GPU 使用率信息
    def print_gpu_usage_config(self):
        for index, usage in self.gpu_usage_config.items():
            print(f"模型 {index} 的GPU使用情况:")
            print(f"  模型加载时内存增加: {usage['model']} MB")
            print(f"  推理过程中的最大内存使用: {usage['infer']} MB")
            print(f"  理论上平均GPU使用率: {usage['average_usage_rate']:.2f}%")

