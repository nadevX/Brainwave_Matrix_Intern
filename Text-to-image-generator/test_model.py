from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
device = "cpu"  # Use "cuda" if GPU is available and supported

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def generate_image(prompt, output_path="generated_image.png"):
    """Generate an image based on a text prompt."""
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Image saved at {output_path}")

if __name__ == "__main__":
    text_prompt = "A serene landscape with mountains and a river during sunset"
    generate_image(text_prompt)
