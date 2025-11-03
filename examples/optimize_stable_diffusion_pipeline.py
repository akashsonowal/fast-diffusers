MODEL = "runwayml/stable-diffusion-v1-5"
VARIANT = None 
CUSTOM_PIPELINE = None 
SCHEDULER = "EulerAncestralDiscreteScheduler"
LORA = None
CONTROLNET = None
STEPS = 30 
PROMPT = 'best quality, realistic, unreal engine, 4K, a beautiful girl'
NEGATIVE_PROMPT = None 
SEED = None 
WARMUPS = 3 
BATCH = 1
HEIGHT = None 
WIDTH = None 
INPUT_IMAGE = None 
CONTROL_IMAGE = None 
OUTPUT_IMAGE = None 
EXTRA_CALL_KWARGS = None 

import sys 
import os 
import importlib 
import inspect 
import argparse
import time 
import json 
import torch 
from pathlib import Path
from PIL import (Image, ImageDraw)
from diffusers.utils import load_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from fast_diffusers.compilers.diffusion_pipeline_compiler import compile, CompilationConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=MODEL)
    parser.add_argument('--variant', type=str, default=VARIANT)
    parser.add_argument('--custom-pipeline', type=str, default=CUSTOM_PIPELINE)
    parser.add_argument('--scheduler', type=str, default=SCHEDULER)
    parser.add_argument('--lora', type=str, default=LORA)
    parser.add_argument('--controlnet', type=str, default=CONTROLNET)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--prompt', type=str, default=PROMPT)
    parser.add_argument('--negative-prompt', type=str, default=NEGATIVE_PROMPT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--warmups', type=int, default=WARMUPS)
    parser.add_argument('--batch', type=int, default=BATCH)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--extra-call-kwargs',
                        type=str,
                        default=EXTRA_CALL_KWARGS)
    parser.add_argument('--input-image', type=str, default=INPUT_IMAGE)
    parser.add_argument('--control-image', type=str, default=CONTROL_IMAGE)
    parser.add_argument('--output-image', type=str, default=OUTPUT_IMAGE)
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--no-fusion', action='store_true')
    return parser.parse_args()

def load_model(pipeline_cls,
               model, 
               variant=None,
               custom_pipeline=None,
               scheduler=None,
               lora=None,
               controlnet=None):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs['cusom_pipeline'] = custom_pipeline
    if variant is not None:
        extra_kwargs['variant'] = variant 
    if controlnet is not None:
        from diffusers import ControlNetModel 
        controlnet = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
        extra_kwargs['controlnet'] = controlnet
    model = pipeline_cls.from_pretrained(model, torch_dtype=torch.float16, **extra_kwargs)

    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module('diffusers'), scheduler)
        model.schdeuler = scheduler_cls.from_config(model.scheduler.config)
    
    if lora is not None:
        model.load_lora_weights(lora)
        model.fuse_lora()
    model.safety_checker = None 
    model.to(torch.device('cuda'))
    return model 

def compile_model(model):
    config = CompilationConfig.Default()

    # xformers and Triton are suggested for achieving best performance.
    # It might be slow for Triton to generate, compile and fine-tune kernels.
    try:
        import xformers 
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    # NOTE:
    # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
    # Disable Triton if you encounter this problem.
    try:
        import triton 
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')
    # NOTE:
    # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
    # My implementation can handle dynamic shape with increased need for GPU memory.
    # But when your GPU VRAM is insufficient or the image resolution is high,
    # CUDA Graph could cause less efficient VRAM utilization and slow down the inference,
    # especially when on Windows or WSL which has the "shared VRAM" mechanism.
    # If you meet problems related to it, you should disable it.
    config.enable_cuda_graph = True

    model = compile(model, config)
    return model
    

def main():
    args = parse_args()
    if args.input_image is None:
        from diffusers import AutoPipelineForText2Image as pipeline_cls
    else:
        from diffusers import AutoPipelineForImage2Image as pipeline_cls
    



if __name__ == "__main__":
    main()