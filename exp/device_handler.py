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
        self.gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES']) if torch.cuda.is_available else None
        self.trainable = trainable

        self.t = time.time()
        self.threshold = random.uniform(30, 60)

    
    def step(self):
        now = time.time()
        switch_flag = now - self.t >= self.threshold

        if self.gpu_id is not None:
            # Switching to GPU
            if switch_flag and self.device == CPU:
                try:
                    used = gpu_info()[self.gpu_id]['mem_used_percent']/100
                    if used < 0.65 and random.random() > used:
                        self._switch(GPU)
                        print("Switched to GPU {}.".format(self.gpu_id))
                        gpu_failed = False
                except RuntimeError as err:
                    if 'cuda' in str(err).lower(): 
                        gpu_failed = True
                    else:
                        raise

            # Switching to CPU
            if (switch_flag and self.device == GPU) or gpu_failed:
                self._switch(CPU)
                if not gpu_failed:
                    print("Switched from GPU {} to CPU.".format(self.gpu_id))
                else:
                    print("Failed to move models to GPU {}. Moved back to CPU.".format(self.gpu_id))
                
        self.t = time.time()
        self.t_threshold = random.uniform(30, 60)


    def forward_manage(self, func):

        def wrapper(b):
            while True:
                forward_failed = False
                try:
                    b = b.to(self.device)
                    v1, v2 = func(b)
                    break
                except RuntimeError as err:
                    if 'cuda' in str(err).lower():
                        forward_failed = True
                    else:
                        raise
                
                if forward_failed:
                    self._switch(CPU)
                    b = b.to(CPU)
                    print("Failed to forward in GPU {}. Moved back to CPU.".format(self.gpu_id))
            return v1, v2

        return wrapper

    def reset(self):
        gc.collect()
        if self.gpu_id is not None:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.device = CPU
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)
        
    def _switch(self, switch_to):
        self.trainable.model = self.trainable.model.to(switch_to)
        self.trainable.opt.state = _recursive_opt_to(switch_to, self.trainable.opt.state)
        gc.collect()
        if switch_to == GPU:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.device = switch_to

        
def _recursive_opt_to(device, var):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = _recursive_opt_to(device, var[key])
        elif torch.is_tensor(var[key]):
            var[key] = var[key].cpu() if device == CPU else var[key].cuda()

    return var

