import time
import random
import torch

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
                switch = used < 0.70 and random.random() > used
                
            if switch:
                model, opt = self.switch_device(model, opt)

        self.t = now
        self.t_threshold = random.uniform(30, 60)
                
        return model, opt

    def switch_device(self, model, opt):
        switch_to = GPU if self.device == CPU else CPU
        try:
            model.to(switch_to)
            opt.state = self._recursive_opt_to(switch_to, opt.state)
            if switch_to == CPU:
                torch.cuda.empty_cache()
            print(f"Switched to {'CPU' if switch_to == CPU else f'GPU {self.gpu_id}'}.")
        except RuntimeError:
            if switch_to == GPU:
                print(f"Failed to switch to GPU {self.gpu_id}. Going back to CPU.")
                switch_to = CPU
                model.to(switch_to)
                opt.state = self._recursive_opt_to(switch_to, opt.state)
                torch.cuda.empty_cache()
            else:
                raise

        self.device = switch_to

        return model, opt

    def reset(self):
        self.device = CPU
        self.t0 = time.time()
        self.t_threshold = random.uniform(30, 60)
        torch.cuda.empty_cache()

    def _recursive_opt_to(self, device, var):
        for key in var:
            if isinstance(var[key], dict):
                var[key] = self._recursive_opt_to(device, var[key])
            elif torch.is_tensor(var[key]):
                var[key] = var[key].to(device)
        
        return var
