from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import io
from PIL import Image
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Enable CORS for all routes (you can restrict this to specific origins in production)
CORS(app)

# Set your Gemini API key as an environment variable
if "GEMINI_API_KEY" not in os.environ:
    raise KeyError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(image_bytes, mime_type=None):
    """Uploads the given image bytes to Gemini."""
    print(f"Preparing to upload image with MIME type: {mime_type}")

    # Check if the MIME type is an image type
    if mime_type not in ['image/jpeg', 'image/png']:
        raise ValueError(f"Unsupported image type: {mime_type}")

    # Use tempfile to create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(image_bytes)
        temp_file_path = temp_file.name
    
    try:
        # Upload the temporary file to Gemini
        file = genai.upload_file(temp_file_path, mime_type=mime_type)
        print(f"Uploaded image as: {file.uri}")
        return file
    finally:
        # Clean up: remove the temporary file after upload
        os.remove(temp_file_path)

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        print("Received request to process image")

        # Check if 'image' is in the request
        if 'image' not in request.files:
            print("No 'image' part in the request")
            return jsonify({'error': 'No image part'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            print("No file selected for upload")
            return jsonify({'error': 'No selected file'}), 400

        # Read image bytes
        image_bytes = image_file.read()
        print(f"Read {len(image_bytes)} bytes from the uploaded image")

        # Check if image is empty
        if len(image_bytes) == 0:
            print("Uploaded image file is empty")
            return jsonify({'error': 'Empty image file'}), 400

        # Upload the image to Gemini
        print("Uploading image to Gemini...")
        uploaded_file = upload_to_gemini(image_bytes, mime_type=image_file.content_type)

        # Create and configure the model
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain",
        }
        print(f"Generation configuration set: {generation_config}")

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",  
            generation_config=generation_config,
        )

        # Generate content from the image
        print("Generating content from image...")
        response = model.generate_content([uploaded_file, "Answer the question in the image."])
        print(f"Received response from model: {response.text}")

        # Return the generated answer
        return jsonify({'answer': response.text})

    except ValueError as ve:
        print(f"Value error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
