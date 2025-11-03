import logging 
import packaging.version
from dataclasses import dataclass
import functools
import torch 
from fast_diffusers.cuda.graphs import (
    make_dynamic_graphed_callable,
    # apply_auto_graph_Compiler,
)
from fast_diffusers.utils import gpu_device
from fast_diffusers.utils.memory_format import apply_memory_format

logger = logging.getLogger()

class CompilationConfig:

    @dataclass 
    class Default:
        memory_format: torch.memory_format = (
            torch.channels_last if gpu_device.device_has_tensor_core() else 
            torch.contiguous_format)
        enable_jit: bool = True 
        enable_jit_freeze: bool = True 
        enable_xformers: bool = False
        enable_cuda_graph: bool = False
        enable_triton: bool = False
        trace_scheduler: bool = False

def compile(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    m.unet = compile_unet(m.unet, config)
    if hasattr(m, 'controlnet'):
        m.controlnet = compile_unet(m.controlnet, config)
    m.vae = compile_vae(m.vae, config)

    if config.enable_jit:
        lazy_trace_ = _build_lazy_trace(config)

        if getattr(m, 'text_encoder', None) is not None:
            m.text_encoder.forward = lazy_trace_(m.text_encoder.forward)
        # for SDXL
        if getattr(m, 'text_encoder_2', None) is not None:
            m.text_encoder_2.forward = lazy_trace_(m.text_encoder_2.forward)
        # for SVD
        if getattr(m, 'image_encoder', None) is not None:
            m.image_encoder.forward = lazy_trace_(m.image_encoder.forward)
        if config.trace_scheduler:
            m.scheduler.scale_model_input = lazy_trace_(
                m.scheduler.scale_model_input)
            m.scheduler.step = lazy_trace_(m.scheduler.step)
    
    if enable_cuda_graph:
        if getattr(m, 'text_encoder', None) is not None:
            m.text_encoder.forward = make_dynamic_graphed_callable(
                m.text_encoder.forward)
        if getattr(m, 'text_encoder_2', None) is not None:
            m.text_encoder_2.forward = make_dynamic_graphed_callable(
                m.text_encoder_2.forward)
        if getattr(m, 'image_encoder', None) is not None:
            m.image_encoder.forward = make_dynamic_graphed_callable(
                m.image_encoder.forward)
    
    if hasattr(m, 'image_processor'):
        from fast_diffusers.libs.diffusers.image_processor import patch_image_processor
        patch_image_processor(m.image_processor)

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
    
    if config.enable_jit:
        lazy_trace_ = _build_lazy_trace(
            config,
            enable_triton_reshape=enable_cuda_graph,
            enable_triton_layer_norm=enable_cuda_graph,
        )
        m.forward = lazy_trace_(m.forward)
    
    if enable_cuda_graph:
        m.forward = make_dynamic_graphed_callable(m.forward)
    
    return m

def compile_vae(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    if config.enable_xformers:
        _enable_xformers(m)
    
    if config.memory_format is not None:
        apply_memory_format(m, memory_format=config.memory_format) 
    return m

def _modify_model():
    pass 

def _ts_compiler():
    pass 

def _build_lazy_trace(config, 
                      enable_triton_reshape=False,
                      enable_triton_layer_norm=False):
    pass 

def _build_ts_compiler(config, 
                       enable_triton_reshape=False,
                       enable_triton_layer_norm=False):
    pass 

def _enable_xformers(m):
    from xformers import ops 

    if hasattr(m, 'enable_xformers_memory_efficient_attention'):
        m.enable_xformers_memory_efficient_attention()
    
    else:
        logger.warning(
            'enable_xformers_memory_efficient_attention() is not available.'
            ' If you have enabled xformers by other means, ignore this warning.'
        )