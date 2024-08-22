from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import gradio as gr


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

    def LunchGradio(txt2img):
        gr.Interface(
                txt2img,
                gr.Text(),
                gr.Image(),
                title='Stable Diffusion2.0 with Gradio UI'
        ).launch(
            share=True,
            debug=True
        )
