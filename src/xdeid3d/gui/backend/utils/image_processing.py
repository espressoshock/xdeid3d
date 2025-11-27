from PIL import Image
import io
from pathlib import Path
from typing import Union


def validate_image(content: bytes) -> bool:
    """Validate if the content is a valid image"""
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
        return True
    except Exception:
        return False


async def save_upload(content: bytes, filepath: Path):
    """Save uploaded file"""
    with open(filepath, "wb") as f:
        f.write(content)


def resize_image(image: Union[Image.Image, bytes], size: tuple = (512, 512)) -> Image.Image:
    """Resize image to target size"""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Crop to square first
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))

    # Resize
    image = image.resize(size, Image.Resampling.LANCZOS)

    return image
