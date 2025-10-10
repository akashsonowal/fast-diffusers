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

# from fast_diffusers.compilers.diffusion_pipeline_compiler import compile, CompilationConfig

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

def load_model(pipeline_cls,
            model,
            variant=None,
            custom_pipeline=None,
            scheduler=None,
            lora=None,
            controlnet=None):
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if controlnet is not None:
        from diffusers import ControlNetModel
        controlnet_model = ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)
        extra_kwargs["controlnet"] = controlnet_model
    
    model = pipeline_cls.from_pretrained(model, torch_dtype=torch.float16, **extra_kwargs)

    if scheduler is not None:
        scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
        model.scheduler = scheduler_cls.from_config(model.scheduler.config)
    
    if lora is not None:
        model.load_lora_weights(lora)
        model.fuse_lora()
    
    model.safety_checker = None 
    model.to(torch.device("cuda"))
    return model

def main():
    args = parse_args()
    if args.input_image is None:
        from diffusers import AutoPipelineForText2Image as pipeline_cls
    else:
        from diffusers import AutoPipelineForImage2Image as pipeline_cls
    
    model = load_model(
        pipeline_cls,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
    )

    height = args.height or model.unet.config.sample_size * model.vae_scale_factor
    width = args.width or model.unet.config.sample_size * model.vae.scale_factor

    if args.quantize:
        
        def quantize_unet(m):
            from diffusers.utils import USE_PEFT_BACKEND
            assert USE_PEFT_BACKEND
            m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
            return m 
        
        model.unet = quantize_unet(model.unet)
        if hasattr(model, "controlnet"):
            model.controlnet = quantize_unet(model.controlnet)
    
    if args.no_fusion:
        torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])
    
    if args.compiler == "none":
        pass 
    elif args.compiler in ("compile", "compile-max-autotune"):
        mode = "max-autotune" if args.compiler == "compile-max-autotune" else None 
        model.unet = torch.compile(model.unet, mode=mode)
        if hasattr(mode, "controlnet"):
            model.controlnet = torch.compile(model.controlnet, mode=mode)
        model.vae = torch.compile(model.vae, mode=mode)
    else:
        raise ValueError(f"Unsupported compiler: {args.compiler}")
    
    if args.input_image is None:
        input_image = None
    else:
        input_image = load_image(args.input_image)
        input_image = input_image.resize((width, height), Image.LANCZOS)
    
    if args.control_image is None:
        if args.controlnet is None:
            control_image = None
        else:
            control_image = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(control_image)
            
        

    def get_kwargs_input():
        kwargs_input = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            num_images_per_prompt=args.batch,
            generator=None if args.seed is None else torch.Generator(device="cuda").manual_seed(args.seed),
            **(dict() if args.extra_call_kwargs is None else json.loads(args.extra_call_kwargs)),
        )
        if input_image is not None:
            kwargs_input["image"] = input_image
        if control_image is not None:
            if input_image is None:
                kwargs_input["image"] = 
