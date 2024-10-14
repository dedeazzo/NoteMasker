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
    reader = easyocr.Reader(['en'])  # Specify language here (e.g., 'en' for English)
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    raise

# Define parameters for processing
DEFAULT_BATCH_SIZE = 5  # Default number of pages per batch
DEFAULT_NUM_WORKERS = min(4, os.cpu_count() // 2)  # Use half of the available CPU cores or a fixed number
DEFAULT_BLUR_RADIUS = 45  # Radius for Gaussian blur
DEFAULT_CORNER_RADIUS = 90  # Radius for rounded corners
DEFAULT_HEIGHT_THRESHOLD_MULTIPLIER = 1.8  # Multiplier for filtering out disproportionately high boxes
DEFAULT_DPI_RESOLUTION = 600  # Resolution for image conversion


def load_input_file(input_file, dpi_resolution=DEFAULT_DPI_RESOLUTION):
    """Load the input file and return a list of pages (as images)."""
    try:
        # Check file extension to decide whether it's a PDF or an image
        if input_file.lower().endswith('.pdf'):
            # Convert PDF to images (pages)
            pages = convert_from_path(input_file, dpi_resolution)
        else:
            # Handle image file directly
            img = Image.open(input_file)
            pages = [img]  # Convert single image into a list of one page
        
        return pages
    except Exception as e:
        print(f"Error loading input file: {e}")
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


def draw_boxes(image, results, height_threshold_multiplier=DEFAULT_HEIGHT_THRESHOLD_MULTIPLIER, blur_radius=DEFAULT_BLUR_RADIUS, corner_radius=DEFAULT_CORNER_RADIUS):
    """Draw blurred bounding boxes with rounded corners, filtering out disproportionately high boxes."""
    try:
        original_image = image.convert("RGBA")
        cv_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGBA2BGRA)
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


def process_page(page, i, height_threshold_multiplier, blur_radius, corner_radius):
    """Function to preprocess, run OCR, and draw boxes on a single page."""
    try:
        input_image = page.convert('RGB')  # Convert the PIL image to RGB format
        preprocessed_image = preprocess_image(input_image)  # Preprocess the image
        results = ocr_image(preprocessed_image)  # Perform OCR
        result_image = draw_boxes(input_image, results, height_threshold_multiplier, blur_radius, corner_radius)  # Draw bounding boxes

        # Save the processed image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file_path = temp_file.name
            result_image.save(temp_file_path)

        return temp_file_path, i
    except Exception as e:
        print(f"Error processing page {i}: {e}")
        return None, i


def main(input_file, output_pdf, batch_size=DEFAULT_BATCH_SIZE, num_workers=DEFAULT_NUM_WORKERS,
         blur_radius=DEFAULT_BLUR_RADIUS, corner_radius=DEFAULT_CORNER_RADIUS,
         height_threshold_multiplier=DEFAULT_HEIGHT_THRESHOLD_MULTIPLIER, dpi_resolution=DEFAULT_DPI_RESOLUTION):
    try:
        # Load the input file (PDF or image)
        pages = load_input_file(input_file, dpi_resolution)

        # Create a new PDF canvas for the output
        c = canvas.Canvas(output_pdf, pagesize=A4)

        # Split pages into batches
        num_batches = math.ceil(len(pages) / batch_size)

        for batch_index in range(num_batches):
            batch_start = batch_index * batch_size
            batch_end = min((batch_index + 1) * batch_size, len(pages))
            page_batch = pages[batch_start:batch_end]

            # Process each batch using multithreading
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_page = {executor.submit(process_page, page, i, height_threshold_multiplier, blur_radius, corner_radius): i for i, page in enumerate(page_batch, start=batch_start)}

                for future in as_completed(future_to_page):
                    try:
                        temp_file_path, i = future.result()
                        if temp_file_path:
                            # Draw the processed image onto the PDF canvas
                            c.drawImage(temp_file_path, 0, 0, width=A4[0], height=A4[1])
                            c.showPage()  # End the current page and start a new one
                            os.remove(temp_file_path)  # Remove the temporary file
                    except Exception as e:
                        print(f"Error processing page {i}: {e}")

        # Save the final PDF output
        c.save()
        print(f"PDF processing completed: {output_pdf}")

    except Exception as e:
        print(f"Error in main processing: {e}")
        raise


# Example usage:
# main('input.pdf', 'output.pdf', batch_size=10, num_workers=4)
