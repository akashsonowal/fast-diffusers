MODEL = "runwayml/stable-diffusion-v1-5"
PROMPT = 'best quality, realistic, unreal engine, 4K, a beautiful girl'

import argparse
from diffusers.utils import load_image

def parse_args():
    parser = argparse.ArgumentParser()

def main():
    args = parse_args()
    if args.input_image is None:
        from diffusers import AutoPipelineForText2Image as pipeline_cls
    else:
        from diffusers import AutoPipelineForImage2Image as pipeline_cls

    