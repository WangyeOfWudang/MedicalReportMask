from fastapi import FastAPI, File, UploadFile,HTTPException
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
import os
import argparse
import json
from collections import defaultdict
from PIL import Image, ImageDraw
import time
import uuid
import re


from surya.input.langs import replace_lang_with_code, get_unique_langs
from surya.input.load import load_from_folder, load_from_file, load_lang_file
from surya.model.detection.segformer import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor
from surya.model.recognition.tokenizer import _tokenize
from surya.ocr import run_ocr
from surya.postprocessing.text import draw_text_on_image
from surya.settings import settings

from pydantic import BaseModel

app = FastAPI()
# 预加载模型和处理器
det_processor = load_detection_processor()
det_model = load_detection_model()
lang_tokens = _tokenize("", get_unique_langs([['zh']])) # 示例: 假设只用英语
rec_model = load_recognition_model(langs=lang_tokens)
rec_processor = load_recognition_processor()

def OCR_Text(input_path, results_dir, max_pages=None, start_page=0, langs=None, lang_file=None):
    assert langs or lang_file, "Must provide either langs or lang_file"

    if os.path.isdir(input_path):
        images, names = load_from_folder(input_path, max_pages, start_page)
        print("images:",images)
        folder_name = os.path.basename(input_path)
    else:
        images, names = load_from_file(input_path, max_pages, start_page)
        print("images:",images)
        folder_name = os.path.basename(input_path).split(".")[0]

    if lang_file:
        # We got all of our language settings from a file
        langs = load_lang_file(lang_file, names)
        for lang in langs:
            replace_lang_with_code(lang)
        image_langs = langs
    else:
        # We got our language settings from the input
        langs = langs.split(",")
        replace_lang_with_code(langs)
        image_langs = [langs] * len(images)
        print("image_langs:",image_langs)


    result_path = results_dir
    os.makedirs(result_path, exist_ok=True)

    predictions_by_image = run_ocr(images, image_langs, det_model, det_processor, rec_model, rec_processor)

    for name, pred, image in zip(names, predictions_by_image, images):
        out_pred = pred.model_dump()
        out_pred["page"] = 1  # Assuming a single page processed for simplicity

        # Save each image's result to a JSON file named after the image
        with open(os.path.join(result_path, f"{name}.json"), "w+", encoding="utf-8") as f:
            json.dump(out_pred, f, ensure_ascii=False)

    print(f"Wrote results to {result_path}")

def load_keywords(keyword_path):
    with open(keyword_path, 'r') as file:
        keywords = [line.strip() for line in file.readlines()]
    return set(keywords)

def apply_mask(image_path, json_data, keywords):
    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # Iterate through each text line and check if the text should be masked
        for item in json_data['text_lines']:
            text = item['text']
            if any(keyword.lower() in text.lower() for keyword in keywords):
                polygon = [tuple(point) for point in item['polygon']]
                draw.polygon(polygon, fill=(255, 255, 255, 255))  # Fill with white
        
        # Save the masked image
        masked_image_path = image_path.replace('.jpeg', '_masked.jpeg')  # Adjust file type if necessary
        img.save(masked_image_path)
        print(f'Masked image saved to {masked_image_path}')
        return masked_image_path

def apply_mask(image_path, json_data, keywords):
    # Open the image
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        
        # Iterate through each text line and check if the text should be masked
        for item in json_data['text_lines']:
            text = item['text']
            # Check if the text contains any keywords or is a number with 8 or more digits
            if any(keyword.lower() in text.lower() for keyword in keywords) or re.match(r'\d{8,}', text):
                polygon = [tuple(point) for point in item['polygon']]
                draw.polygon(polygon, fill=(255, 255, 255, 255))  # Fill with white
        
        # Save the masked image
        masked_image_path = image_path.replace('.jpeg', '_masked.jpeg')  # Adjust file type if necessary
        img.save(masked_image_path)
        print(f'Masked image saved to {masked_image_path}')
        return masked_image_path


class ImageData(BaseModel):
    base64_image: str

def decode_base64_image(data_string):
    image_data = base64.b64decode(data_string)
    return Image.open(BytesIO(image_data))

def encode_image_to_base64(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/process-image/")
async def process_image(data: ImageData):
    request_id = str(uuid.uuid4())
    temp_image_path = f"./temp/{request_id}.jpeg"
    results_dir = f"./results/{request_id}"
    os.makedirs(results_dir, exist_ok=True)

    image = decode_base64_image(data.base64_image)
    image.save(temp_image_path)
    
    # 使用您提供的 OCR_Text 函数处理图像
    results_dir = "./results"
    OCR_Text(input_path=temp_image_path, results_dir=results_dir, max_pages=None, start_page=0, langs="zh", lang_file=None)
    
    keyword_path = './keywords.txt'  # 加载关键词
    keywords = load_keywords(keyword_path)
    
    # 假设 JSON 数据保存在固定路径
    json_path = os.path.join(results_dir, f"{request_id}.json")
    json_folder = os.path.join(results_dir, f"{request_id}")
    json_data = json.load(open(json_path, 'r'))
    
    # 应用遮罩并返回结果
    masked_image_path = apply_mask(temp_image_path, json_data, keywords)
    masked_image = Image.open(masked_image_path)
    base64_output = encode_image_to_base64(masked_image)
    
    # Cleanup temporary files
    os.remove(temp_image_path)
    os.remove(masked_image_path)
    os.remove(json_path)
    os.rmdir(json_folder)

    return JSONResponse(content={"image": base64_output})
