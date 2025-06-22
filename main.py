from m2 import EnhancedMultimodalDocumentProcessor
from enhanced_document_query import EnhancedMultimodalQuerySystem
import os
import jpype
from docx import Document
from fpdf import FPDF
import pandas as pd
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
import jpype
import numpy as np
from flask import send_file, request, jsonify,Flask
from dotenv import load_dotenv
load_dotenv()
from urllib.parse import unquote
import mimetypes


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize JVM
jvm_path = r"C:\Program Files\Java\jdk-24\bin\server\jvm.dll"
if not jpype.isJVMStarted():
    jpype.startJVM(jvm_path, "--enable-native-access=ALL-UNNAMED")

# Initialize managers
GEMINI_API_KEY = os.getenv("Gemini")
doc_manager = EnhancedMultimodalDocumentProcessor(gemini_api_key=GEMINI_API_KEY)
query_manager = EnhancedMultimodalQuerySystem(gemini_api_key=GEMINI_API_KEY)

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj

@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """API endpoint to upload and process a document"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        custom_name = request.form.get('custom_name')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        file_ext = filename.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as temp_input:
            file.save(temp_input.name)
            original_path = temp_input.name

        # Convert to PDF if needed
        pdf_filename = f"{Path(filename).stem}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)

        if file_ext == 'docx':
            document = Document(original_path)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", size=12)
            for para in document.paragraphs:
                pdf.multi_cell(0, 10, para.text)
            pdf.output(pdf_path)

        elif file_ext in ['xlsx', 'csv']:
            df = pd.read_excel(original_path) if file_ext == 'xlsx' else pd.read_csv(original_path)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            col_width = pdf.w / (len(df.columns) + 1)
            row_height = pdf.font_size + 2

            for col in df.columns:
                pdf.cell(col_width, row_height, str(col), border=1)
            pdf.ln(row_height)

            for _, row in df.iterrows():
                for item in row:
                    pdf.cell(col_width, row_height, str(item), border=1)
                pdf.ln(row_height)
            pdf.output(pdf_path)

        elif file_ext == 'pdf':
            pdf_path = original_path  # No conversion needed

        else:
            os.remove(original_path)
            return jsonify({'error': 'Unsupported file type'}), 400

        # Process the final PDF
        result = doc_manager.process_document(pdf_path, custom_name)

        # Clean up
        if os.path.exists(original_path) and original_path != pdf_path:
            os.remove(original_path)
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        return convert_numpy(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """API endpoint to list all processed documents"""
    try:
        documents = doc_manager.list_processed_documents()
        return jsonify({
            'status': 'success',
            'documents': documents,
            'total_count': len(documents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """API endpoint to delete a processed document"""
    try:
        result = doc_manager.delete_processed_document(doc_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_documents():
    """API endpoint to query documents"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query_text = data.get('query')
        doc_ids = data.get('doc_ids', [])
        top_k = data.get('top_k', 3)
        
        if not query_text:
            return jsonify({'error': 'Query text is required'}), 400
        
        if not doc_ids:
            return jsonify({'error': 'At least one document ID is required'}), 400
        
        # Perform query
        result = query_manager.query_specific_document(query_text=query_text,doc_id= doc_ids, top_k=top_k)
        
        return jsonify(result['answer'])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 


@app.route('/api/get-image', methods=['GET'])
def get_image():
    """API endpoint to serve images from file paths"""
    try:
        # Get the image path from query parameters
        image_path = request.args.get('path')
        
        if not image_path:
            return jsonify({'error': 'No image path provided'}), 400
        
        # URL decode the path
        decoded_path = unquote(image_path)
        
        # Security check: ensure the path is safe and within allowed directories
        # Adjust this path validation based on your document storage structure
        if not decoded_path.startswith('document_storage'):
            return jsonify({'error': 'Invalid path: Access denied'}), 403
        
        # Check if file exists
        if not os.path.exists(decoded_path):
            return jsonify({'error': f'File not found: {decoded_path}'}), 404
        
        # Check if it's actually a file (not a directory)
        if not os.path.isfile(decoded_path):
            return jsonify({'error': 'Path is not a file'}), 400
        
        # Get the mime type
        mime_type, _ = mimetypes.guess_type(decoded_path)
        
        # Ensure it's an image file
        if not mime_type or not mime_type.startswith('image/'):
            return jsonify({'error': 'File is not an image'}), 400
        
        # Serve the file
        return send_file(
            decoded_path,
            mimetype=mime_type,
            as_attachment=False,
            download_name=os.path.basename(decoded_path)
        )
        
    except Exception as e:
        return jsonify({'error': f'Error serving image: {str(e)}'}), 500

@app.route('/api/get-file-info', methods=['GET'])
def get_file_info():
    """API endpoint to get file information"""
    try:
        file_path = request.args.get('path')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        
        decoded_path = unquote(file_path)
        
        # Security check
        if not decoded_path.startswith('document_storage'):
            return jsonify({'error': 'Invalid path: Access denied'}), 403
        
        if not os.path.exists(decoded_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Get file information
        stat = os.stat(decoded_path)
        mime_type, _ = mimetypes.guess_type(decoded_path)
        
        file_info = {
            'filename': os.path.basename(decoded_path),
            'size': stat.st_size,
            'mime_type': mime_type,
            'last_modified': stat.st_mtime,
            'is_image': mime_type and mime_type.startswith('image/') if mime_type else False,
            'exists': True
        }
        
        return jsonify(file_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Optional: Endpoint to list images in a document directory
@app.route('/api/list-document-images/<doc_id>', methods=['GET'])
def list_document_images(doc_id):
    """API endpoint to list all images for a specific document"""
    try:
        # Construct the document's image directory path
        doc_images_dir = os.path.join('document_storage', doc_id, 'images')
        
        if not os.path.exists(doc_images_dir):
            return jsonify({
                'doc_id': doc_id,
                'images': [],
                'message': 'No images directory found'
            })
        
        images = []
        for filename in os.listdir(doc_images_dir):
            filepath = os.path.join(doc_images_dir, filename)
            if os.path.isfile(filepath):
                mime_type, _ = mimetypes.guess_type(filepath)
                if mime_type and mime_type.startswith('image/'):
                    stat = os.stat(filepath)
                    images.append({
                        'filename': filename,
                        'path': filepath,
                        'size': stat.st_size,
                        'mime_type': mime_type
                    })
        
        return jsonify({
            'doc_id': doc_id,
            'images': images,
            'total_count': len(images)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
