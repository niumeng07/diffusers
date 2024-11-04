from diffusers import DiffusionPipeline

# 模型目录，如果不存在则自动下载
pipe = DiffusionPipeline.from_pretrained("../models/stable-diffusion-3.5-large-turbo")
pipe = pipe.to("cuda")  # 使用 GPU 加速

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

name_prefix = base64.b64encode(prompt.encode("utf-8"))
# decode
# base64.b64decode(imagename_prefix)
for indx, image in enumerate(pipe(prompt).images):
    image.save(f"{name_prefix}_{indx}.png")
