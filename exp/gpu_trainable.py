import os
import time
import random
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import gc
from nvgpu import gpu_info

import torch
from ray import tune
from exp.train_wrapper import TrainWrapper

CPU = 'cpu'
GPU = 'cuda'

class GPUTrainable(tune.Trainable):
    
    def setup(self, config):
        self.device = CPU
        self.usage_threshold = config['gpu_threshold']
        
        self.gpu_ids = config['gpu_ids']
        self.curr_gpu = random.choice(config['gpu_ids']) if config['gpu_ids'] else None
        if self.gpu_ids:
            torch.cuda.set_device(self.curr_gpu)
        
        self.t = time.time()
        self.threshold = random.uniform(10, 90)

        self.train_wrap = TrainWrapper(config)

    def step(self):
        while True:
            forward_failed = False
            try:
                res_dict = self.train_wrap.step('cpu' if self.device == CPU else f'cuda:{self.curr_gpu}')
                break
            except RuntimeError as e:
                print(e)
                str_e = str(e).lower()
                if 'cuda' in str_e or 'cudnn' in str_e: 
                    forward_failed = True
                else:
                    raise
            
            if forward_failed:
                print(f"Failed forward in GPU {self.curr_gpu}. Moved back to CPU.")
                self.device = self._switch(CPU)

        if self.gpu_ids and self.device != GPU:
            self._attempt_switch()

        return res_dict

    def _attempt_switch(self):  
        now = time.time()
        if now - self.t >= self.threshold:
            gpus_usage = gpu_info()

            # Selecting the GPU with minimum memory usage
            if len(self.gpu_ids) > 1 and random.random() > 0.5:
                min_memory = min([gpus_usage[i]['mem_used_percent'] for i in self.gpu_ids])
                gpu_indices = list(filter(lambda el: gpus_usage[el]['mem_used_percent'] == min_memory, self.gpu_ids))
                
                if self.curr_gpu not in gpu_indices:
                    self.curr_gpu = random.choice(gpu_indices)
                    torch.cuda.set_device(self.curr_gpu)

            # Switching to GPU
            used = gpus_usage[self.curr_gpu]['mem_used_percent']/100
            if used < self.usage_threshold and random.random() > used/self.usage_threshold:
                self.device = self._switch(GPU) 
            else:
                self.t_threshold = random.uniform(10, 30)
                self.t = time.time()

    def _switch(self, device):
        if device == GPU:
            gpu_failed = False
            try:
                self.train_wrap.model = self.train_wrap.model.cuda()
                self.train_wrap.opt.state = _recursive_opt_to(GPU, self.train_wrap.opt.state)
                print(f"Switched to GPU {self.curr_gpu}.")
            except RuntimeError as e:
                str_e = str(e).lower()
                if 'cuda' in str_e or 'cudnn' in str_e: 
                    gpu_failed = True
                else:
                    raise
            
            if not gpu_failed:
                return GPU
            else:
                print(f"Failed switch to GPU {self.curr_gpu}. Moved back to CPU.")
                self.t = time.time()
                self.t_threshold = random.uniform(30, 90)
                return self._switch(CPU) 
        elif device == CPU:
            self.train_wrap.model = self.train_wrap.model.cpu()
            self.train_wrap.opt.state = _recursive_opt_to(CPU, self.train_wrap.opt.state)
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            return CPU

    def save_checkpoint(self, tmp_checkpoint_dir):
        torch.save(self.train_wrap.model.state_dict(), os.path.join(tmp_checkpoint_dir, "model.pth"))
        torch.save(self.train_wrap.opt.state_dict(), os.path.join(tmp_checkpoint_dir, "opt.pth"))
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        mod_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "model.pth"), map_location='cpu')
        opt_state_dict = torch.load(os.path.join(tmp_checkpoint_dir, "opt.pth"), map_location='cpu')
        self.train_wrap.model.load_state_dict(mod_state_dict)
        self.train_wrap.opt.load_state_dict(opt_state_dict)

    def cleanup(self):
        del self.train_wrap.model, self.train_wrap.opt
        gc.collect()
        if self.gpu_ids:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


def _recursive_opt_to(device, var):
    for key in var:
        if isinstance(var[key], dict):
            var[key] = _recursive_opt_to(device, var[key])
        elif torch.is_tensor(var[key]):
            var[key] = var[key].cpu() if device == CPU else var[key].cuda()

    return var
