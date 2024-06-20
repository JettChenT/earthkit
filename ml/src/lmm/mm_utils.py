# Utility functions for multimodal LLM calls
from PIL import Image, ImageDraw, ImageFont
import os
from io import BytesIO
import base64

def render_text_description(
        image: Image.Image, text: str, line_height=16
) -> Image.Image:
    """
    Render a text description at the bottom of an image. Assumes that text is single line.
    """
    # Create a new image with a black background
    bar = Image.new("RGB", (image.width, int(line_height * 1.2)), "black")

    # Create a draw object and add text to the bar
    draw = ImageDraw.Draw(bar)
    font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "Inter-Regular.ttf")
    font = ImageFont.truetype(font_path, line_height)
    text_width = draw.textlength(text, font)
    position = ((bar.width - text_width) / 2, int(line_height * 0.1))
    draw.text(position, text, fill="white", font=font)

    # Concatenate the original image with the bar
    image_with_bar = Image.new("RGB", (image.width, image.height + bar.height))
    image_with_bar.paste(image, (0, 0))
    image_with_bar.paste(bar, (0, image.height))

    return image_with_bar

def encode_image(image: Image.Image, max_size_mb=10):

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert max_size_mb to bytes
    max_encoded_size_bytes = (
        (max_size_mb * 1024 * 1024) * 3 / 4 if max_size_mb else None
    )

    # Initial setup for quality
    quality = 100

    while True:
        virtual_file = BytesIO()
        image.save(virtual_file, format="JPEG", quality=quality)
        img_data = virtual_file.getvalue()
        encoded_data = base64.b64encode(img_data).decode("utf-8")

        # Check if the encoded data size is within the specified limit
        if (
                max_encoded_size_bytes is None
                or len(encoded_data) <= max_encoded_size_bytes
        ):
            break
        # If not, decrease quality
        print("compressing image")
        quality -= 5
        if quality <= 10:  # Prevent the quality from becoming too low
            break

    return f"data:image/jpeg;base64,{encoded_data}"