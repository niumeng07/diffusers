from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("../models/stable-diffusion-3.5-large-turbo")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

for indx, image in enumerate(pipe(prompt).images):
    image.save(f"{indx}.png")
