import logging 
import packaging.version
from dataclasses import dataclass
import functools
import torch 

logger = logging.getLogger()

class CompilationConfig:

    @dataclass 
    class Default:
        memory_format: torch.memory_format = (
            
        )
        enable_xformers: bool = False
        enable_cuda_graph: bool = False
        enable_triton: bool = False

def compile(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    m.unet = compile_unet(m.unet, config)
    return m

def compile_unet(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    if config.enable_xformers:
        _enable_xformers(m)
    
    if config.memory_format is not None:
        apply_memory_format(m, memory_format=config.memory_format)

def _enable_xformers(m):
    from xformers import ops 

    if hasattr(m, 'enable_xformers_memory_efficient_attention'):
        m.enable_xformers_memory_efficient_attention()
    
    else:
        logger.warning(
            'enable_xformers_memory_efficient_attention() is not available.'
            ' If you have enabled xformers by other means, ignore this warning.'
        )