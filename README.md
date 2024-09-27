# PDF Processor with OCR and Blurring

This project is a Python-based utility for processing PDF files. It converts PDF pages to images, applies OCR (Optical Character Recognition) using [EasyOCR](https://github.com/JaidedAI/EasyOCR), and blurs text within bounding boxes meant to redact text and leave diagrams for creating exercises from notes.

## Features
- **PDF to Image Conversion**: Uses `pdf2image` to convert PDF pages into high-resolution images.
- **OCR**: Uses EasyOCR to extract text from images.
- **Bounding Box Detection**: Merges bounding boxes and filters out oversized boxes to refine text area detection.
- **Blurring Sensitive Text**: Blurs regions where text appears, using rounded rectangular masks to preserve aesthetics.
- **Multithreading**: Utilizes multithreading to process pages in parallel for faster execution.
- **Output**: Generates a new PDF with processed images where sensitive information is blurred out.

## Requirements

Before you begin, make sure you have Python 3.x installed on your system. Install the required Python packages by running:

```bash
pip install pdf2image Pillow opencv-python-headless numpy easyocr reportlab
```

Additionally, make sure you have the following system dependencies:

- **poppler-utils**: Required for converting PDF to images.
- **tesseract-ocr**: For OCR functionality (optional, as this project uses EasyOCR, but may be useful for comparison).

For Ubuntu, install these dependencies using:

```bash
sudo apt install poppler-utils
```

For macOS:

```bash
brew install poppler
```

## Project Structure

- **Main Script**: The main processing logic is within the `main()` function, which orchestrates PDF-to-image conversion, OCR, bounding box detection, and blurring.
- **Functions**:
  - `preprocess_image()`: Converts images to grayscale, applies binary thresholding, and removes noise.
  - `ocr_image()`: Performs OCR on preprocessed images.
  - `merge_boxes()`: Merges close bounding boxes to prevent overlapping detections.
  - `calculate_average_height()`: Computes the average height of detected bounding boxes to filter out irrelevant ones.
  - `draw_boxes()`: Draws blurred bounding boxes on detected text regions.
  - `process_page()`: Processes each page by preprocessing, applying OCR, and drawing blurred boxes.
  
## How to Use

1. Place your input PDF file (e.g., `Bones.pdf`) in the root directory.
2. Adjust the `input_pdf` and `pdf_path` variables in the script if necessary. The `input_pdf` is the source PDF, and `pdf_path` is the output PDF path.

Example:
```python
input_pdf = 'Bones.pdf'
pdf_path = 'Bones_processed.pdf'
```
3. Run the script:
```bash
python main.py
```

### Batch Size and Worker Configuration

You can adjust the batch size and the number of worker threads used in processing:

```python
BATCH_SIZE = 5  # Number of pages processed per batch
NUM_WORKERS = min(4, os.cpu_count() // 2)  # Number of parallel threads for processing
```

### Error Handling
- Errors during OCR initialization or processing are caught and logged for troubleshooting.
- Temporary image files created during processing are automatically cleaned up after each page is processed.

### Example Workflow

1. The PDF is split into individual pages and converted into images.
2. Each image is preprocessed to enhance text visibility.
3. OCR is performed on each page to detect text regions.
4. Bounding boxes are merged and blurred to obscure sensitive content.
5. A new PDF is generated with processed pages.

### Multithreading and Performance

The code uses the `ThreadPoolExecutor` to parallelize the processing of PDF pages. The `NUM_WORKERS` parameter controls how many threads are used simultaneously, leveraging multicore processors for improved performance.

## Customization

- **Thresholds**: You can adjust the binary thresholding and morphological operations in the `preprocess_image()` function to fine-tune how the images are preprocessed.
- **Bounding Box Filtering**: The `height_threshold_multiplier` in the `draw_boxes()` function determines which boxes to ignore based on height.

## Output

After successful execution, a new PDF (e.g., `Bones_processed.pdf`) will be generated with sensitive text blurred.

## Troubleshooting

1. **EasyOCR Installation Issues**: If EasyOCR fails to initialize, check if all the required model files are downloaded properly.
2. **Performance Bottlenecks**: For large PDFs, you might want to decrease `BATCH_SIZE` to process fewer pages in each batch.
3. **PDF Conversion Errors**: Ensure you have `poppler-utils` installed correctly if PDF to image conversion fails.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
