# Enhanced Document Query System

A Streamlit-based application that allows users to upload, manage, and query documents using Google Gemini AI for intelligent document processing and natural language querying.

## 🚀 Features

- **Document Upload & Management**: Upload multiple document formats (PDF, DOCX, TXT, etc.)
- **Intelligent Document Processing**: Automatic text extraction and chunking using advanced processing
- **Interactive Document Selection**: Choose single or multiple documents via dropdown interface
- **Gemini AI Integration**: Leverage Google's Gemini AI for sophisticated document querying
- **Vector Storage**: Efficient document storage with embeddings for fast retrieval
- **Multimodal Support**: Handle text, images, and tables within documents
- **Session Management**: Persistent document state across user interactions
- **User-Friendly Interface**: Clean, intuitive Streamlit interface with enhanced UI components

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager
- Internet connection (for AI model APIs)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd multimodal_task
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Required Dependencies

Your `requirements.txt` should include:

```
streamlit>=1.28.0
google-generativeai>=0.3.0
langchain>=0.1.0
langchain-google-genai>=0.0.5
chromadb>=0.4.0
sentence-transformers>=2.2.0
pandas>=1.5.0
numpy>=1.24.0
python-docx>=0.8.11
PyPDF2>=3.0.0
Pillow>=9.0.0
python-dotenv>=1.0.0
streamlit-extras>=0.3.0
```

## ⚙️ Configuration

1. **Environment Variables**
   
   Create a `.env` file in the root directory:
   ```env
   Gemini=your_google_gemini_api_key_here
   # Optional: Add other service API keys if needed
   ```

2. **Google AI Studio Setup**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key for Gemini
   - Copy the API key to your `.env` file

3. **Configuration Settings**
   - Document processing parameters in `processor.py`
   - Query engine settings in `m2.py`
   - UI configurations in `main.py`

## 🚀 How to Run

1. **Start the application**
- First Terminal
   ```bash
   streamlit run ui.py
   ```
- Second Terminal
   ```bash
   python main.py
   ```

2. **Access the application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The application will load automatically

3. **Alternative UI Components**
   ```bash
   # If you want to run specific components
   streamlit run ui.py          # Alternative UI

   ```

## 📖 Usage Guide

### Step 1: Upload Documents
1. Click on the file uploader section in the main interface
2. Select one or more documents (PDF, DOCX, TXT supported)
3. Wait for the document processing to complete (handled by `m2.py`)
4. Documents will be automatically chunked and stored with embeddings

### Step 2: Select Documents for Querying
1. Use the enhanced dropdown menu to select documents
2. Choose single or multiple documents from your processed collection
3. Use "Select All" or "Clear All" buttons for quick actions
4. Selected documents will be highlighted in the interface

### Step 3: Ask Questions with Gemini AI
1. Enter your question in the text input field
2. Click "Submit" or press Enter
3. Gemini AI (via `processor.py`) will analyze your query
4. View intelligent responses with context from selected documents

### Step 4: Review Results
- Responses include relevant excerpts from selected documents
- Source documents and chunks are cited for transparency
- Multimodal content (images, tables) is referenced when relevant
- Follow-up questions can be asked for deeper exploration

### Advanced Features
- **Multimodal Processing**: Images and tables are extracted and processed
- **Intelligent Chunking**: Documents are split into meaningful segments
- **Vector Search**: Fast semantic search across document embeddings
- **Session Persistence**: Your document selections and history are maintained

## 🏗️ Project Structure

```
enhanced-document-query/
├── main.py                           # Main Streamlit application
├── m2.py                            # Document processing and chunking
├── processor.py                     # Gemini AI query processing engine
├── ui.py                            # Enhanced UI components
├── llm.py                           # Language model utilities
├── retriever.py                     # Document retrieval system
├── enhanced_document_query.py       # Core document query logic
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (create this)
├── .gitignore                       # Git ignore file
├── README.md                        # This file
├── document_storage/                # Document storage directory
│   ├── Guidelines_for_making_MC.../  # Processed document folders
│   │   ├── images/                  # Extracted images
│   │   ├── tables/                  # Extracted tables
│   │   ├── text_chunks/             # Text chunks
│   │   ├── chunks.pkl               # Serialized chunks
│   │   ├── embeddings.pkl           # Document embeddings
│   │   └── subdirs.json             # Directory metadata
│   ├── sample_multimodal_data2_b.../# Additional processed documents
│   └── documents_metadata.json      # Global document metadata
├── multimodel_venv/                 # Virtual environment (optional)
├── uploads/                         # Temporary uploaded files
└── sample_multimodal_data2.pdf      # Sample document for testing
```

## 🔧 Customization

### Modifying Document Processing (`m2.py`)
1. Adjust chunk sizes and overlap parameters
2. Modify text extraction methods for different file types
3. Configure embedding generation settings
4. Customize multimodal content extraction

### Enhancing Query Processing (`processor.py`)
1. Update Gemini AI model parameters
2. Modify prompt templates for better responses
3. Adjust response formatting and citations
4. Add custom query preprocessing logic

### UI & API Enhancements (`ui.py`, `main.py`)
1. Customize the Streamlit interface layout
2. Add new interactive components
3. Modify styling with custom CSS
4. Integrate additional visualization features

### Adding New AI Models (`llm.py`)
1. Integrate additional language models
2. Create model comparison interfaces
3. Add fallback model options
4. Implement model-specific optimizations

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Google API key errors**
   - Verify your `.env` file exists and contains a valid `Gemini`
   - Check API key permissions in Google AI Studio
   - Ensure Gemini API is enabled for your project

3. **Document processing failures (`m2.py`)**
   - Check if document format is supported
   - Verify file is not corrupted or password-protected
   - Ensure sufficient disk space for processing
   - Check file size limits (adjust in code if needed)

4. **Gemini AI query issues (`processor.py`)**
   - Verify internet connection
   - Check API quotas and rate limits
   - Ensure document chunks are properly formatted
   - Review query complexity and length

5. **Slow response times**
   - Check internet connection stability
   - Consider reducing document chunk sizes in `m2.py`
   - Optimize embedding generation parameters
   - Clear document storage cache periodically

6. **Storage issues (`document_storage/`)**
   - Ensure write permissions for the directory
   - Check available disk space
   - Clear old processed documents if needed
   - Verify pickle file integrity

### Performance Optimization

- **Large Documents**: Adjust chunking parameters in `m2.py`
- **Multiple Documents**: Implement batch processing optimizations
- **Memory Usage**: Clear session state and temporary files regularly
- **Response Speed**: Cache frequent queries and embeddings
- **Storage Management**: Implement automatic cleanup of old documents

## 📊 Supported File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Text extraction supported |
| Word | `.docx` | Full document structure |
| Text | `.txt` | Plain text files |
| Markdown | `.md` | Formatted text |
| CSV | `.csv` | Tabular data |

## 🔒 Security Considerations

- **API Keys**: Never commit API keys to version control
- **File Uploads**: Validate file types and sizes
- **Data Privacy**: Consider local processing for sensitive documents
- **Session Management**: Clear sensitive data after sessions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: Check the Wiki for detailed guides
- **Community**: Join our Discord/Slack for discussions


## 🙏 Acknowledgments

- Streamlit community for the excellent framework
- OpenAI/Anthropic for AI model APIs
- LangChain for document processing utilities
- Contributors and beta testers

---

**Made with ❤️ by [Your Name/Team]**

For more information, contact us at [komalgupta991000@gmail.com]