# File Redaction & OCR Application - README

## Overview

This application is a Flask-based web service designed to process PDFs and images. It uses Optical Character Recognition (OCR) via EasyOCR to detect text in files, applies redaction by blurring specific text areas within bounding boxes, and generates a processed PDF. The purpose is to redact notes while retaining diagrams for creating exercises, ensuring that sensitive or unwanted text is removed.

The app supports concurrent multi-page processing using multithreading, making it efficient for handling large documents.

## Features

- **Supports PDFs and images** (`.pdf`, `.png`, `.jpg`, `.jpeg`)
- **OCR for Text Detection** using EasyOCR.
- **Text Redaction**: Blurs text in bounding boxes while leaving other content (e.g., diagrams) intact.
- **User-customizable Parameters**: Allows adjustment of batch size, blur intensity, corner radius, etc.
- **Multithreading**: Processes pages in batches for efficient large file handling.
- **Downloadable Output**: Generates a processed PDF with redacted text that can be downloaded by the user.

## Installation

### Prerequisites
Make sure you have the following installed:
- Python 3.6+
- pip (Python package installer)

### Step 1: Clone the repository
```bash
git clone https://github.com/dedeazzo/NoteMasker.git
cd redaction-ocr-app
```

### Step 2: Install required packages
```bash
pip install -r requirements.txt
```

### Step 3: Install additional dependencies
You will need to install the following manually:
- **EasyOCR**: Optical Character Recognition
- **OpenCV**: Image processing
- **ReportLab**: PDF generation
- **Pillow**: Image manipulation
- **pdf2image**: Convert PDF pages to images

To install these:
```bash
pip install easyocr opencv-python-headless reportlab Pillow pdf2image
```

### Step 4: Additional setup for PDFs
On some systems, `pdf2image` requires `poppler-utils`. Install via:
- **Ubuntu**: `sudo apt-get install poppler-utils`
- **MacOS**: `brew install poppler`

### Step 5: Set up upload and processed folders
Ensure the following directories exist:
```bash
mkdir uploads
mkdir processed
```

### Step 6: Run the Flask Application
```bash
python app.py
```

The application will run on [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

## Usage

### 1. Upload File
- Open the app in your browser.
- Drag and drop a PDF or image into the upload area or select a file manually.
  
### 2. Configure Settings
- Click the hamburger menu at the top right to open settings.
- Adjust the parameters:
  - **Batch Size**: Number of pages processed concurrently.
  - **Number of Workers**: Number of CPU cores to utilize.
  - **Blur Radius**: Intensity of the blur effect for redaction.
  - **Corner Radius**: Radius for rounded corners of redaction boxes.
  - **Height Threshold Multiplier**: Controls the size of bounding boxes.
  - **DPI Resolution**: Image resolution for PDF-to-image conversion.

### 3. Processing and Download
- After uploading, the file is processed, and once completed, a download link for the redacted PDF appears.
- Click the link to download the processed PDF.

## API Endpoints

### 1. `GET /`
Renders the file upload page.

### 2. `POST /upload`
Uploads a file and processes it based on the configured settings.
- Request Parameters (form data):
  - `file`: The PDF or image file to be uploaded.
  - `batch_size`: (optional) The number of pages to process in a batch.
  - `num_workers`: (optional) Number of CPU cores to use for concurrent processing.
  - `blur_radius`: (optional) Radius for the blur effect (default: 45).
  - `corner_radius`: (optional) Radius for rounded corners on redaction boxes.
  - `height_multiplier`: (optional) Multiplier for filtering bounding boxes.
  - `dpi_resolution`: (optional) DPI for converting PDFs to images (default: 600).
  
- Response:
  - Returns a JSON object with the filename of the processed PDF.

### 3. `GET /download/<filename>`
Downloads the processed PDF file.

## Customization

You can adjust the default processing parameters in `file_scrubber.py`:
- **Batch Size**: Controls how many pages are processed together.
- **Blur Radius**: The amount of blur applied to the text.
- **Corner Radius**: The smoothness of redaction box corners.
- **Height Threshold Multiplier**: Controls whether large bounding boxes are included.

Modify these parameters in the `main()` function or through the web interface.

## Known Issues

- **Performance**: Processing large files with many pages can be slow, depending on system resources.
- **OCR Accuracy**: EasyOCR may not detect all text accurately, especially for images with poor quality or complex fonts.
  
## Contributing

Feel free to submit issues or contribute to the project by opening a pull request.

## License

This project is licensed under the MIT License.