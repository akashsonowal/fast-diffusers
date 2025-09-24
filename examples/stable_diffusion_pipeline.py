import time
import torch 

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

def load_model():
    model = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16)
    model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        model.scheduler.config)
    model.safety_checker = None 
    model.to(torch.device("cuda"))
    return model 

model = load_model()
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

kwargs_inputs = dict(
    prompt='(masterpiece:1,2), best quality, masterpiece, best detailed face, a beautiful girl',
    height=512,
    width=512,
    num_inference_steps=30,
    num_images_per_prompt=1,
)

for _ in range(3):
    output_image = model(**kwargs_inputs).images[0]

begin = time.time()
output_image = model(**kwargs_inputs).images[0]
print(f'Inference time: {time.time() - begin:.3f}s') # 1.928 seconds