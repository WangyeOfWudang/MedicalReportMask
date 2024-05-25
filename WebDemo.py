import gradio as gr
import requests
import base64
from PIL import Image
import numpy as np
import io

def process_image(image):
    # 检查图像数据类型并转换为 PIL.Image 对象（如果需要）
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'), 'RGB')

    # 将图片转换为base64编码
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 使用本地地址调用 FastAPI
    url = 'http://127.0.0.1:8000/process-image/'

    # 发送请求
    response = requests.post(url, json={"base64_image": img_str})
    if response.status_code == 200:
        # 处理返回的图片
        data = response.json()
        image_data = base64.b64decode(data['image'])
        result_image = Image.open(io.BytesIO(image_data))
        return result_image
    else:
        return "Error: Failed to process image."

# 设置 Gradio 界面
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(label="Original Image"),
    outputs=gr.Image(label="Processed Image"),
    title="Image Processing Demo",
    description="Upload an image to apply masking."
)

# 启动 Gradio 应用，启用调试模式和共享
iface.launch(share=True)
