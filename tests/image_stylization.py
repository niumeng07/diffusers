import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("../models/stable-diffusion-3.5-large-turbo")
pipe = pipe.to("cuda")  # 使用 GPU 加速

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    return image.resize((new_width, new_height))

init_image = load_image("path/to/your/image.jpg")

prompt = "A beautiful painting of a sunset in the style of Van Gogh, oil painting, highly detailed, vibrant colors"
strength = 0.75  # 控制初始图像和生成图像之间的相似度，值在 0~1 之间
guidance_scale = 7.5  # 引导系数，值越高图像越符合 prompt 描述

with torch.autocast("cuda"):
    stylized_image = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]

stylized_image.show()  # 显示风格化后的图像
stylized_image.save("stylized_image.jpg")  # 保存风格化后的图像
