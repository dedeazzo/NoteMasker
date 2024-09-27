from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import cv2
import math
import easyocr
import tempfile
import numpy as np

# Initialize EasyOCR Reader
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    raise

# Paths to the input PDF and output PDF
input_pdf = 'Bones.pdf'
pdf_path = 'Bones_processed.pdf'

# Define a reasonable batch size and the number of worker threads
BATCH_SIZE = 5  # Process 5 pages per batch
NUM_WORKERS = min(4, os.cpu_count() // 2)  # Use half of the available CPU cores or a fixed number

try:
    # Convert PDF to images
    pages = convert_from_path(input_pdf, 600)
except Exception as e:
    print(f"Error converting PDF to images: {e}")
    raise

def preprocess_image(image):
    """Preprocess the image to enhance text visibility."""
    try:
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert image to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Denoise the image using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        return cleaned_image
    except cv2.error as e:
        print(f"Error processing image: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during preprocessing: {e}")
        raise

def ocr_image(image):
    """Perform OCR on the preprocessed image."""
    try:
        # Convert the processed image to a format EasyOCR can handle
        pil_image = Image.fromarray(image)
        results = reader.readtext(np.array(pil_image))
        return results
    except Exception as e:
        print(f"OCR failed: {e}")
        raise

def merge_boxes(boxes, threshold=10):
    """Merge bounding boxes that are close to each other into one larger box."""
    try:
        merged_boxes = []
        for bbox, _, _ in boxes:
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1

            if merged_boxes and (x1 <= merged_boxes[-1][2] + threshold and y1 <= merged_boxes[-1][3] + threshold):
                last_box = merged_boxes.pop()
                new_x1 = min(last_box[0], x1)
                new_y1 = min(last_box[1], y1)
                new_x2 = max(last_box[2], x2)
                new_y2 = max(last_box[3], y2)
                merged_boxes.append((new_x1, new_y1, new_x2, new_y2))
            else:
                merged_boxes.append((x1, y1, x2, y2))
        
        return [(box, '', 0) for box in merged_boxes]  # Return in the same format as `results`
    except Exception as e:
        print(f"Error merging boxes: {e}")
        raise

def calculate_average_height(results):
    """Calculate the average height of the bounding boxes."""
    try:
        heights = []
        for bbox, _, _ in results:
            top_left = tuple(map(int, bbox[0]))
            bottom_left = tuple(map(int, bbox[3]))
            height = abs(bottom_left[1] - top_left[1])
            heights.append(height)

        if heights:
            return sum(heights) / len(heights)
        return 0
    except Exception as e:
        print(f"Error calculating average height: {e}")
        raise

def draw_boxes(image, results, height_threshold_multiplier=1.8):
    """Draw blurred bounding boxes with rounded corners, filtering out disproportionately high boxes."""
    try:
        # Convert PIL image to RGBA mode
        original_image = image.convert("RGBA")
        
        # Convert the image to OpenCV format for blurring
        cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGBA2BGRA)

        blur_radius = 45
        corner_radius = 90
        avg_height = calculate_average_height(results)

        for (bbox, _, _) in reversed(results):
            top_left = tuple(map(int, bbox[0]))
            top_right = tuple(map(int, bbox[1]))
            bottom_right = tuple(map(int, bbox[2]))
            bottom_left = tuple(map(int, bbox[3]))

            box_height = abs(bottom_left[1] - top_left[1])
            if box_height > avg_height * height_threshold_multiplier:
                continue

            mask = Image.new("L", original_image.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            x1, y1 = min(top_left[0], bottom_left[0]), min(top_left[1], top_right[1])
            x2, y2 = max(bottom_right[0], top_right[0]), max(bottom_right[1], bottom_left[1])
            mask_draw.rounded_rectangle([x1, y1, x2, y2], radius=corner_radius, fill=255)

            roi = cv_image[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (0, 0), blur_radius)
            cv_image[y1:y2, x1:x2] = blurred_roi

            blurred_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))
            original_image.paste(blurred_image, mask=mask)
        
        return original_image
    except UnidentifiedImageError as e:
        print(f"Error identifying image: {e}")
        raise
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        raise

def process_page(page, i):
    """Function to preprocess, run OCR, and draw boxes on a single page."""
    try:
        # Convert the PIL image to RGB format
        input_image = page.convert('RGB')

        # Preprocess the image
        preprocessed_image = preprocess_image(input_image)

        # Perform OCR
        results = ocr_image(preprocessed_image)

        # Draw bounding boxes and underlines on the original image
        result_image = draw_boxes(input_image, results)

        # Save the processed image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file_path = temp_file.name
            result_image.save(temp_file_path)

        return temp_file_path, i

    except Exception as e:
        print(f"Error processing page {i}: {e}")
        return None, i

def main():
    try:
        # Create a new PDF canvas
        c = canvas.Canvas(pdf_path, pagesize=A4)

        # Split pages into batches
        num_batches = math.ceil(len(pages) / BATCH_SIZE)

        for batch_index in range(num_batches):
            batch_start = batch_index * BATCH_SIZE
            batch_end = min((batch_index + 1) * BATCH_SIZE, len(pages))
            page_batch = pages[batch_start:batch_end]

            # Process each batch using multithreading
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_page = {executor.submit(process_page, page, i): i for i, page in enumerate(page_batch, start=batch_start)}

                for future in as_completed(future_to_page):
                    try:
                        temp_file_path, i = future.result()
                        if temp_file_path:
                            # Draw the image on the PDF
                            c.drawImage(temp_file_path, 0, 0, width=A4[0], height=A4[1])
                            c.showPage()  # End the current page and start a new one
                            # Clean up the temporary file
                            os.remove(temp_file_path)

                    except Exception as e:
                        print(f"Error processing future for page {i}: {e}")
                        continue  # Skip this page if there's an error

        c.save()  # Save the PDF at the end

    except Exception as e:
        print(f"Error saving PDF: {e}")
        raise

if __name__ == "__main__":
    main()
