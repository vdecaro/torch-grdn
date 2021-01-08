import os
import time
import random
import torch

import gc
from nvgpu import gpu_info

CPU = 'cpu'
GPU = 'cuda:0'

class DeviceHandler(object):
    
    def __init__(self, trainable):
        self.device = CPU
        self.gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES']) if torch.cuda.is_available() else None
        self.trainable = trainable

        self.t = time.time()
        self.threshold = random.uniform(0, 30)

    
    def step(self):
        if self.gpu_id is not None and self.device != GPU:
            # Switching to GPU
            gpu_failed = False
            now = time.time()
            used = gpu_info()[self.gpu_id]['mem_used_percent']/100
            switch_flag = now - self.t >= self.threshold and used < 0.65 and random.random() > used/0.65
            if switch_flag:
                try:
                    self._switch(GPU)
                    print("Switched to GPU {}.".format(self.gpu_id))
                except RuntimeError as e:
                    str_e = str(e).lower()
                    if 'cuda' in str_e or 'cudnn' in str_e: 
                        gpu_failed = True
                    else:
                        raise
            
                # Switching to CPU
                if gpu_failed:
                    self._switch(CPU)
                    if not gpu_failed:
                        print("Switched from GPU {} to CPU.".format(self.gpu_id))
                    else:
                        print("Failed to move models to GPU {}. Moved back to CPU.".format(self.gpu_id))
                    self.t_threshold = random.uniform(30, 90)
                else:
                    self.t_threshold = random.uniform(10, 30)
                    
                self.t = time.time()

    def forward_manage(self, func):

        def wrapper(b):
            while True:
                forward_failed = False
                try:
                    b = b.to(self.device)
                    v1, v2 = func(b)
                    break
                except RuntimeError as e:
                    str_e = str(e).lower()
                    if 'cuda' in str_e or 'cudnn' in str_e: 
                        forward_failed = True
                    else:
                        raise
                
                if forward_failed:
                    self._switch(CPU)
                    b = b.to(CPU)
                    self.t = time.time()
                    self.t_threshold = random.uniform(30, 90)
                    print("Failed to forward in GPU {}. Moved back to CPU.".format(self.gpu_id))
            return v1, v2

        return wrapper

    def reset(self):
        gc.collect()
        if self.gpu_id is not None:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        self.device = CPU
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)
        
    def _switch(self, switch_to):
        self.trainable.model = self.trainable.model.to(switch_to)
        self.trainable.opt.state = _recursive_opt_to(switch_to, self.trainable.opt.state)
        gc.collect()
        if switch_to == CPU:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        self.device = switch_to

        
def _recursive_opt_to(device, var):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = _recursive_opt_to(device, var[key])
        elif torch.is_tensor(var[key]):
            var[key] = var[key].cpu() if device == CPU else var[key].cuda()

    return var
