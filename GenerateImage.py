from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


class GenerateImage():
    def makepipe(model_id="stabilityai/stable-diffusion-2-1"):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=EulerDiscreteScheduler.from_pretrained(
                model_id,
                subfolder="scheduler"
            ),
            torch_dtype=torch.float16).to("cuda")
        return pipe

    def txt2img(pipe, txt):
        image = pipe(txt, height=128, width=128).images
        return image
