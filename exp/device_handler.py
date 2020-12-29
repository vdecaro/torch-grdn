import time
import random
import torch

import gc
from nvgpu import gpu_info

CPU = 'cpu'
GPU = 'cuda:0'

class DeviceHandler(object):
    
    def __init__(self, gpu_id):
        self.device = CPU
        self.gpu_id = gpu_id

        self.t = time.time()
        self.threshold = random.uniform(30, 60)

    
    def step(self, model, opt):
        now = time.time()
        switch_to = GPU if self.device == CPU else CPU
        if now - self.t >= self.threshold:
            switch = True
            if switch_to == GPU:
                used = gpu_info()[self.gpu_id]['mem_used_percent']/100
                switch = used < 0.65 and random.random() > used

            if switch:
                to_cpu = False
                try:
                    model, opt = self.switch(switch_to, model, opt)
                    if switch_to == GPU:
                        print("Switched to GPU {}.".format(self.gpu_id))
                    if switch_to == CPU:
                        print("Switched to CPU from GPU {}.".format(self.gpu_id))
                except RuntimeError:
                    if switch_to == GPU:
                        to_cpu = True
                    else:
                        raise
                if to_cpu:
                    model, opt = self.switch(CPU, model, opt)
                    print("Failed to move models to GPU {}. Moved back to CPU.".format(self.gpu_id))
                    self.gpu_mem_reset()
                
        self.t = time.time()
        self.t_threshold = random.uniform(30, 60)
                
        return model, opt
    
    def forward_manage(func):

        def wrapper(b):
            while True:
                to_cpu = False
                try:
                    if torch.cuda.is_available() and device == 'cuda:0':
                        b = b.to(device)
                    v1, v2 = func(b)
                    break
                except RuntimeError as err:
                    if 'cuda' in err.lower():
                        to_cpu = True
                    else:
                        raise
                if to_cpu:
                    self._switch('cpu', self.model, self.opt)
                    print("Failed to forward in GPU {}. Moved back to CPU.".format(self.device_handler.gpu_id))
                    self.gpu_mem_reset()
            return v1, v2

        return wrapper

    def reset(self):
        self.gpu_mem_reset()
        self.device = CPU
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)
        
    def _switch(self, switch_to, model, opt):
        model = model.to(switch_to)
        opt.state = _recursive_opt_to(switch_to, opt.state)
        self.gpu_mem_reset()
        self.device = switch_to
        
        return model, opt
    
    def gpu_mem_reset(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        
        
        
        
        

def _recursive_opt_to(self, device, var):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = self._recursive_opt_to(device, var[key])
        elif torch.is_tensor(var[key]):
            var[key] = var[key].cpu() if device == CPU else var[key].cuda()

    return var

