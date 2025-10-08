MODEL = "runwayml/stable-diffusion-v1-5"
VARIANT = None 
CUSTOM_PIPELINE = None 
SCHEDULER = "EulerAncestralDiscreteScheduler"
LORA = None
CONROLNET = None
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

import importlib 
import inspect 
import argparse
import time 
import json 
import torch 
from PIL import Image, ImageDraw
from diffusers.utils import load_image

from fast_diffusers.compilers.diffusion_pipeline_compiler import compile, CompilationConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--variant", type=str, default=VARIANT)
    parser.add_argument("--custom_pipeline", type=str, default=CUSTOM_PIPELINE)
    parser.add_argument("--scheduler", type=str, default=SCHEDULER)
    parser.add_argument("--lora", type=str, default=LORA)
    parser.add_argument("--controlnet", type=str, default=CONROLNET)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmups", type=int, default=WARMUPS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--extra-call-kwargs", type=str, default=EXTRA_CALL_KWARGS)
    parser.add_argument("--input_image", type=str, default=INPUT_IMAGE)
    parser.add_argument("--control_image", type=str, default=CONTROL_IMAGE)
    parser.add_argument("--output_image", type=str, default=OUTPUT_IMAGE)
    parser.add_argument("--compiler", type=str, default="compile", choices=["none", "fast-diffusers", "compile", "compile-max-autotune"])
    parser.add_argument("--quantize", action="store_true")
    

def main():
    args = parse_args()
    if args.input_image is None:
        from diffusers import AutoPipelineForText2Image as pipeline_cls
    else:
        from diffusers import AutoPipelineForImage2Image as pipeline_cls

    