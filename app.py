from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from file_scrubber import main  # Import the main function from your script

# The directory where uploaded files will be saved
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Check if the uploaded file is allowed (only PDFs or images)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route with the file upload form
@app.route('/')
def index():
    return render_template('index.html')

# Handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    # Extract the file and parameters from the request
    uploaded_file = request.files.get('file')
    
    if uploaded_file:
        # Get parameters from the request
        batch_size = int(request.form.get('batch_size', 10))  # Default value
        num_workers = int(request.form.get('num_workers', 4))  # Default value
        blur_radius = int(request.form.get('blur_radius', 45))  # Default value
        corner_radius = int(request.form.get('corner_radius', 90))  # Default value
        height_threshold_multiplier = float(request.form.get('height_multiplier', 1.8))  # Default value
        dpi_resolution = int(request.form.get('dpi_resolution', 600))  # Default value

        # Save the uploaded file and call your main function with parameters
        input_file_path = f'uploads/{uploaded_file.filename}'
        uploaded_file.save(input_file_path)

        # Correctly construct the output filename
        base_name = uploaded_file.filename.rsplit('.', 1)[0]  # Get the name without the extension
        output_pdf = f'processed/processed_{base_name}.pdf'  # Prepend "processed_" to the base name

        try:
            main(input_file_path, output_pdf, batch_size, num_workers, 
                 blur_radius, corner_radius, height_threshold_multiplier, dpi_resolution)
            return jsonify({'filename': f'processed_{base_name}.pdf'})  # Return the new filename
        except Exception as e:
            return jsonify({'message': str(e)}), 500

    return jsonify({'message': 'No file uploaded.'}), 400

# Route to handle downloading the processed file
@app.route('/download/<filename>')
def download_file(filename):
    try:
        # Send the processed file from the 'processed' directory for download
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)
