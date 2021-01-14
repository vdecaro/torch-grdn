import os
import time
import random
import torch

import gc
from nvgpu import gpu_info

CPU = 'cpu'
GPU = 'cuda'

class DeviceHandler(object):
    
    def __init__(self, trainable, gpu_ids):
        self.trainable = trainable
        self.device = CPU
        
        self.gpu_ids = gpu_ids
        self.curr_gpu = random.choice(gpu_ids) if gpu_ids else None
        torch.cuda.set_device(self.curr_gpu)
        
        self.t = time.time()
        self.threshold = random.uniform(0, 30)

    
    def step(self):
        if self.gpu_ids and self.device != GPU:
            
            now = time.time()
            if now - self.t >= self.threshold:
                
                # Selecting the GPU with minimum memory usage
                if len(self.gpu_ids) > 1 and random.random() > 0.5:
                    gpus_usage = gpu_info()
                    min_idx, min_usage = [], 1000
                    for i in self.gpu_ids:
                        i_usage = gpus_usage[i]['mem_used_percent']
                        if i_usage < min_usage:
                            min_idx, min_usage = [i], i_usage
                        elif i_usage == min_usage:
                            min_idx.append(i)
                            
                    if self.curr_gpu not in min_idx:
                        self.curr_gpu = random.choice(min_idx)
                        torch.cuda.set_device(self.curr_gpu)

                # Switching to GPU
                gpu_failed = False
                used = gpu_info()[self.curr_gpu]['mem_used_percent']/100
                switch_flag = used < 0.65 and random.random() > used/0.65
                if switch_flag:
                    try:
                        self._switch(GPU)
                        print("Switched to GPU {}.".format(self.curr_gpu))
                    except RuntimeError as e:
                        str_e = str(e).lower()
                        if 'cuda' in str_e or 'cudnn' in str_e: 
                            gpu_failed = True
                        else:
                            raise

                    # Switching to CPU
                    if gpu_failed:
                        self._switch(CPU)
                        print("Failed to move models to GPU {}. Moved back to CPU.".format(self.curr_gpu))
                        self.t_threshold = random.uniform(30, 90)
                        self.t = time.time()
                else:
                    self.t_threshold = random.uniform(10, 30)
                    self.t = time.time()

    def forward_manage(self, func):

        def wrapper(b):
            
            while True:
                forward_failed = False
                try:
                    b = b.cuda() if self.device == GPU else b
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
                    b = b.cpu()
                    self.t = time.time()
                    self.t_threshold = random.uniform(30, 90)
                    print("Failed to forward in GPU {}. Moved back to CPU.".format(self.curr_gpu))
                    
            return v1, v2

        return wrapper

    def cleanup(self):
        del self.trainable.model, self.trainable.opt
        gc.collect()
        if self.gpu_ids:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        
    def _switch(self, device):
        self.trainable.model = self.trainable.model.cpu() if device == CPU else self.trainable.model.cuda()
        self.trainable.opt.state = _recursive_opt_to(device, self.trainable.opt.state)
        gc.collect()
        if device == CPU:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        self.device = device


def _recursive_opt_to(device, var):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = _recursive_opt_to(device, var[key])
        elif torch.is_tensor(var[key]):
            var[key] = var[key].cpu() if device == CPU else var[key].cuda()

    return var
