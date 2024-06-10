# for visualize
import base64
from PIL import Image, ImageDraw
import numpy as np
import io
import cv2

def calculate_pixel_area(mask_data):
    areas = []
    for item in mask_data:
        mask_bytes = base64.b64decode(item['mask'])
        mask_image = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_image)
        pixel_count = np.count_nonzero(mask_array)

        areas.append({'label': item['label'], 'pixels': pixel_count})
    return areas


def overlay_mask_on_image(mask_data, index, original_image_path, color, alpha=0.5):
    # Load the original image
    original_image = Image.open(original_image_path).convert("RGBA")

    # Decode and process each mask
    item = mask_data[index]

    # Decode the base64 string to bytes for the mask
    mask_bytes = base64.b64decode(item['mask'])
    mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Convert mask image to numpy array
    mask_array = np.array(mask_image)

    # Create an RGBA image for the mask
    mask_rgba = Image.new("RGBA", mask_image.size)
    for y in range(mask_image.height):
        for x in range(mask_image.width):
            if mask_array[y, x] > 0:  # If the pixel is part of the mask
                mask_rgba.putpixel((x, y), color + (int(255 * alpha),))

    # Combine the original image with the mask
    combined = Image.alpha_composite(original_image, mask_rgba)

    # Draw the border around the mask area
    draw = ImageDraw.Draw(combined)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = contour.squeeze()
        if contour.ndim == 2:
            contour = [tuple(pt) for pt in contour]
            draw.line(contour + [contour[0]], fill=color + (255,), width=3)
    
    original_image = combined

    return original_image

################################################
# for query
import requests
from dotenv import load_dotenv
import os

def query(filename):
    "query to huggingface inference api"

    load_dotenv(dotenv_path = ".env")
    API_URL = os.getenv('API_URL')
    HF_token = os.getenv('HF_token')
    headers = {"Authorization": f"Bearer {HF_token}"}

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)

    return response.json()