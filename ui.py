import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any
import time
import os
import io
from PIL import Image
# Configuration
API_BASE_URL = "http://localhost:5000/api"

# Page configuration
st.set_page_config(
    page_title="Document Query System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin-bottom: 2rem;
    }
    .document-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .query-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict[str, Any]:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            error_msg = response.json().get('error', 'Unknown error') if response.text else f"HTTP {response.status_code}"
            return {"success": False, "error": error_msg}
            
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API server. Make sure the Flask app is running on localhost:5000"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def upload_document(file, custom_name: str = None) -> Dict[str, Any]:
    """Upload document to API"""
    files = {'file': (file.name, file.getvalue(), file.type)}
    data = {}
    if custom_name:
        data['custom_name'] = custom_name
    else:
        data['custom_name'] = file.name
    
    return make_api_request("/upload-document", "POST", data=data, files=files)

def get_documents() -> Dict[str, Any]:
    """Get list of all documents"""
    return make_api_request("/documents")

def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a document"""
    return make_api_request(f"/documents/{doc_id}", "DELETE")

def load_image_from_api(image_path: str) -> Dict[str, Any]:
    """Load image from API endpoint"""
    try:
        # URL encode the path to handle spaces and special characters
        import urllib.parse
        encoded_path = urllib.parse.quote(image_path, safe='')
        url = f"{API_BASE_URL}/get-image?path={encoded_path}"
        
        response = requests.get(url)
        if response.status_code == 200:
            return {"success": True, "content": response.content}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_image_from_path(image_path: str, caption: str = None):
    """Display image from file path using API"""
    result = load_image_from_api(image_path)
    
    if result["success"]:
        try:
            image = Image.open(io.BytesIO(result["content"]))
            st.image(image, caption=caption or f"Image: {os.path.basename(image_path)}", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error displaying image: {str(e)}")
    else:
        st.error(f"❌ Failed to load image: {result['error']}")
        st.code(f"Path: {image_path}")

def format_enhanced_answer(answer_data):
    """Format the enhanced answer response with images/tables"""
    if isinstance(answer_data, list) and len(answer_data) > 0:
        # Handle list of answer objects
        for i, item in enumerate(answer_data):
            if isinstance(item, dict):
                # Display the answer
                if 'Answer' in item:
                    st.markdown("#### 💡 Answer:")
                    st.markdown(item['Answer'])
                
                # Display associated images/tables
                if 'paths' in item and item['paths']:
                    st.markdown("#### 🖼️ Associated Images/Tables:")
                    
                    # Create columns for multiple images
                    if len(item['paths']) > 1:
                        cols = st.columns(min(len(item['paths']), 3))  # Max 3 columns
                        for idx, path in enumerate(item['paths']):
                            with cols[idx % 3]:
                                display_image_from_path(path, f"Reference {idx + 1}")
                    else:
                        # Single image, full width
                        display_image_from_path(item['paths'][0], "Reference Image")
                    
                    # # Show file paths for reference
                    # with st.expander("📁 File Paths"):
                    #     for idx, path in enumerate(item['paths'], 1):
                    #         st.code(f"{idx}. {path}")
                
                if i < len(answer_data) - 1:
                    st.markdown("---")
    
    elif isinstance(answer_data, dict):
        # Handle single answer object
        if 'Answer' in answer_data:
            st.markdown("#### 💡 Answer:")
            st.markdown(answer_data['Answer'])
        
        if 'paths' in answer_data and answer_data['paths']:
            st.markdown("#### 🖼️ Associated Images/Tables:")
            
            if len(answer_data['paths']) > 1:
                cols = st.columns(min(len(answer_data['paths']), 3))
                for idx, path in enumerate(answer_data['paths']):
                    with cols[idx % 3]:
                        display_image_from_path(path, f"Reference {idx + 1}")
            else:
                display_image_from_path(answer_data['paths'][0], "Reference Image")
            
            with st.expander("📁 File Paths"):
                for idx, path in enumerate(answer_data['paths'], 1):
                    st.code(f"{idx}. {path}")
    
def query_documents(query_text: str, doc_ids: List[str], top_k: int = 3) -> Dict[str, Any]:
    """Query documents"""
    data = {
        "query": query_text,
        "doc_ids": doc_ids,
        "top_k": top_k
    }
    return make_api_request("/query", "POST", data=data)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Document Query System</h1>
        <p>Upload documents and query them with natural language</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'selected_docs' not in st.session_state:
        st.session_state.selected_docs = []

    # Sidebar for navigation
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio("Choose Action:", ["Upload Documents", "Manage Documents", "Query Documents"])
        
        st.markdown("---")
        st.markdown("### 🔧 Settings")
        auto_refresh = st.checkbox("Auto-refresh documents", value=True)
        
        if st.button("🔄 Refresh Documents"):
            st.session_state.documents = []

    # Main content based on selected page
    if page == "Upload Documents":
        st.header("📤 Upload Documents")
        
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                uploaded_files = st.file_uploader(
                    "Choose files to upload",
                    accept_multiple_files=True,
                    type=['pdf', 'txt', 'doc', 'docx'],
                    help="Supported formats: PDF, TXT, DOC, DOCX"
                )
            
            with col2:
                custom_name = st.text_input(
                    "Custom Name (Optional)",
                    placeholder="Enter custom document name",
                    help="Leave empty to use original filename"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_files:
                st.write(f"📄 **Selected {len(uploaded_files)} file(s):**")
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size} bytes)")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Upload All Documents", type="primary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        success_count = 0
                        total_files = len(uploaded_files)
                        
                        for i, file in enumerate(uploaded_files):
                            status_text.text(f"Uploading {file.name}...")
                            
                            # Use custom name only if there's one file
                            name_to_use = custom_name if len(uploaded_files) == 1 and custom_name else None
                            result = upload_document(file, name_to_use)
                            
                            if result["success"]:
                                success_count += 1
                                st.success(f"✅ {file.name} uploaded successfully!")
                            else:
                                st.error(f"❌ Failed to upload {file.name}: {result['error']}")
                            
                            progress_bar.progress((i + 1) / total_files)
                        
                        status_text.text(f"Upload complete! {success_count}/{total_files} files successful.")
                        
                        if success_count > 0:
                            st.session_state.documents = []  # Clear cache to refresh
                            time.sleep(1)
                            st.rerun()

    elif page == "Manage Documents":
        st.header("📋 Manage Documents")
        
        # Load documents if not already loaded or if auto-refresh is enabled
        if not st.session_state.documents or auto_refresh:
            with st.spinner("Loading documents..."):
                result = get_documents()
                if result["success"]:
                    st.session_state.documents = result["data"]["documents"]
                else:
                    st.error(f"Failed to load documents: {result['error']}")
                    st.session_state.documents = []
        
        if st.session_state.documents:
            st.success(f"📊 **Total Documents: {len(st.session_state.documents)}**")
            
            # Document cards display
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"📄 {doc.get('doc_name', 'Unknown Document')}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("### 📋 Document Information")
                        st.markdown(f"**📝 Name:** {doc.get('doc_name', 'N/A')}")
                        st.markdown(f"**🆔 Document ID:** `{doc.get('doc_id', 'N/A')}`")
                        st.markdown(f"**📅 Processed Date:** {doc.get('processed_date', 'N/A')}")
                        st.markdown(f"**💾 File Size:** {doc.get('file_size', 'N/A'):,} bytes ({doc.get('file_size', 0) / 1024 / 1024:.2f} MB)")
                    
                    with col2:
                        # Processing Statistics
                        if 'processing_stats' in doc:
                            stats = doc['processing_stats']
                            st.markdown("### 📊 Processing Stats")
                            
                            # Key metrics in a compact format
                            st.metric("📄 Total Pages", stats.get('total_pages', 0))
                            st.metric("🧩 Total Chunks", stats.get('total_chunks', 0))
                            st.metric("📝 Total Characters", f"{stats.get('total_characters', 0):,}")
                    
                    # Detailed statistics in columns
                    if 'processing_stats' in doc:
                        stats = doc['processing_stats']
                        st.markdown("### 📈 Detailed Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "📊 Avg Chunk Size", 
                                f"{stats.get('avg_chunk_size', 0):.0f} chars"
                            )
                        
                        with col2:
                            st.metric(
                                "📑 Chunks per Page", 
                                f"{stats.get('avg_chunks_per_page', 0):.1f}"
                            )
                        
                        with col3:
                            st.metric(
                                "🖼️ Chunks with Images", 
                                stats.get('chunks_with_images', 0)
                            )
                        
                        with col4:
                            st.metric(
                                "📊 Chunks with Tables", 
                                stats.get('chunks_with_tables', 0)
                            )
                        
                        # Progress bars for visual representation
                        if stats.get('total_chunks', 0) > 0:
                            st.markdown("#### 📊 Content Distribution")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                image_percentage = (stats.get('chunks_with_images', 0) / stats.get('total_chunks', 1)) * 100
                                st.markdown(f"**Images Coverage:** {image_percentage:.1f}%")
                                st.progress(image_percentage / 100)
                                
                            with col2:
                                table_percentage = (stats.get('chunks_with_tables', 0) / stats.get('total_chunks', 1)) * 100
                                st.markdown(f"**Tables Coverage:** {table_percentage:.1f}%")
                                st.progress(table_percentage / 100)
            
            # Summary table for quick overview
            st.markdown("### 📋 Quick Overview")
            df_data = []
            for doc in st.session_state.documents:
                stats = doc.get('processing_stats', {})
                df_data.append({
                    "Document Name": doc.get('doc_name', 'Unknown'),
                    "Doc ID": doc.get('doc_id', 'N/A')[:30] + "..." if len(doc.get('doc_id', '')) > 30 else doc.get('doc_id', 'N/A'),
                    "Size (MB)": f"{doc.get('file_size', 0) / 1024 / 1024:.2f}",
                    "Pages": stats.get('total_pages', 0),
                    "Chunks": stats.get('total_chunks', 0),
                    "Images": stats.get('chunks_with_images', 0),
                    "Tables": stats.get('chunks_with_tables', 0),
                    "Processed Date": doc.get('processed_date', 'N/A')[:10] if doc.get('processed_date') else 'N/A'
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Document management section
            st.markdown("### 🗑️ Delete Documents")
            doc_to_delete = st.selectbox(
                "Select document to delete:",
                options=[""] + [f"{doc['doc_name']} (ID: {doc['doc_id'][:20]}...)" for doc in st.session_state.documents],
                help="Choose a document to permanently delete"
            )
            
            if doc_to_delete:
                # Extract the full doc_id from the selected document
                selected_doc_name = doc_to_delete.split(" (ID: ")[0]
                doc_id = None
                for doc in st.session_state.documents:
                    if doc['doc_name'] == selected_doc_name:
                        doc_id = doc['doc_id']
                        break
                
                if doc_id:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        if st.button("🗑️ Delete Document", type="secondary", use_container_width=True):
                            if st.session_state.get('confirm_delete') != doc_id:
                                st.session_state.confirm_delete = doc_id
                                st.warning("⚠️ Click again to confirm deletion")
                            else:
                                result = delete_document(doc_id)
                                if result["success"]:
                                    st.success("✅ Document deleted successfully!")
                                    st.session_state.documents = []  # Clear cache
                                    del st.session_state.confirm_delete
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete document: {result['error']}")
        else:
            st.info("📭 No documents found. Upload some documents first!")

    else:  # Query Documents
        st.header("🔍 Query Documents")
        
        # Load documents if not already loaded
        if not st.session_state.documents:
            with st.spinner("Loading documents..."):
                result = get_documents()
                if result["success"]:
                    st.session_state.documents = result["data"]["documents"]
                else:
                    st.error(f"Failed to load documents: {result['error']}")
                    st.session_state.documents = []
        if st.session_state.documents:
            st.markdown("### 📖 Select Documents to Query")
            
            # Create document name to ID mapping
            doc_names = [f"📄 {doc['doc_name']} (ID: {doc['doc_id'][:20]}...)" for doc in st.session_state.documents]
            doc_id_map = {f"📄 {doc['doc_name']} (ID: {doc['doc_id'][:20]}...)": doc['doc_id'] 
                        for doc in st.session_state.documents}
            
            # Get currently selected document names for default values
            current_selected_names = []
            if hasattr(st.session_state, 'selected_docs') and st.session_state.selected_docs:
                for doc_id in st.session_state.selected_docs:
                    for doc in st.session_state.documents:
                        if doc['doc_id'] == doc_id:
                            current_selected_names.append(f"📄 {doc['doc_name']} (ID: {doc['doc_id'][:20]}...)")
                            break
            
            # Multiselect dropdown for document selection
            selected_doc_names = st.multiselect(
                "Choose documents to query:",
                options=doc_names,
                default=current_selected_names,
                placeholder="Select one or more documents...",
                help="You can select multiple documents to query simultaneously"
            )
            
            # Convert selected names back to doc_ids
            selected_docs = [doc_id_map[name] for name in selected_doc_names]
            st.session_state.selected_docs = selected_docs
            
            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", help="Select all available documents"):
                    st.session_state.selected_docs = [doc["doc_id"] for doc in st.session_state.documents]
                    st.rerun()
            with col2:
                if st.button("Clear All", help="Clear all selections"):
                    st.session_state.selected_docs = []
                    st.rerun()
            
            # Display current selection status
            if st.session_state.selected_docs:
                st.success(f"✅ {len(st.session_state.selected_docs)} document(s) selected")
            else:
                st.info("ℹ️ No documents selected. Please select at least one document to proceed.")
                    
            st.session_state.selected_docs = selected_docs
            
            if selected_docs:
                # st.success(f"✅ Selected {len(selected_docs)} document(s)")
                
                # Query section
                # st.markdown('<div class="query-section">', unsafe_allow_html=True)
                st.markdown("### 💬 Ask a Question")
                
                # Query mode selection
                query_mode = st.radio(
                    "Query Mode:",
                    ["Single Document", "Multiple Documents"],
                    help="Choose whether to query one document at a time or multiple documents together"
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    query_text = st.text_area(
                        "Enter your question:",
                        placeholder="What would you like to know about the selected documents?",
                        height=100
                    )
                
                with col2:
                    top_k = st.slider("Results to return:", min_value=1, max_value=10, value=3)
                    st.markdown("<br>", unsafe_allow_html=True)
                    query_button = st.button("🔍 Query Documents", type="primary", use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if query_button and query_text.strip():
                    with st.spinner("🔍 Searching documents..."):
                        if query_mode == "Single Document":
                            # Query each document individually
                            all_results = []
                            for doc_id in selected_docs:
                                doc_name = next((doc['doc_name'] for doc in st.session_state.documents if doc['doc_id'] == doc_id), doc_id)
                                
                                # Query single document
                                result = query_documents(query_text, doc_id, top_k)
                                
                                if result["success"]:
                                    result_data = result["data"]
                                    all_results.append({
                                        "doc_id": doc_id,
                                        "doc_name": doc_name,
                                        "result": result_data
                                    })
                                else:
                                    st.error(f"❌ Failed to query {doc_name}: {result['error']}")
                            
                            # Display results from all documents
                            if all_results:
                                st.markdown("### 📋 Query Results")
                                
                                for i, doc_result in enumerate(all_results, 1):
                                    with st.expander(f"📄 Results from: {doc_result['doc_name']}", expanded=len(all_results) == 1):
                                        answer_data = doc_result["result"]
                                        
                                        # Use the enhanced formatting function
                                        format_enhanced_answer(answer_data)
                        
                        else:  # Multiple Documents mode
                            # For multiple documents, we'll query them one by one and combine results
                            st.info("ℹ️ Multiple document mode: Querying each document individually and combining results...")
                            
                            all_answers = []
                            all_sources = []
                            
                            for doc_id in selected_docs:
                                doc_name = next((doc['doc_name'] for doc in st.session_state.documents if doc['doc_id'] == doc_id), doc_id)
                                
                                result = query_documents(query_text, doc_id, top_k)
                                
                                if result["success"]:
                                    result_data = result["data"]
                                    if isinstance(result_data, dict):
                                        if "answer" in result_data:
                                            all_answers.append(f"**From {doc_name}:** {result_data['answer']}")
                                        if "sources" in result_data and result_data["sources"]:
                                            for source in result_data["sources"]:
                                                source["from_document"] = doc_name
                                                all_sources.append(source)
                                    elif isinstance(result_data, list):
                                        # Handle new format with Answer and paths
                                        for item in result_data:
                                            if isinstance(item, dict) and 'Answer' in item:
                                                all_answers.append(f"**From {doc_name}:** {item['Answer']}")
                                                if 'paths' in item:
                                                    for path in item['paths']:
                                                        all_sources.append({
                                                            "content": f"Visual content from {os.path.basename(path)}",
                                                            "path": path,
                                                            "from_document": doc_name
                                                        })
                                    else:
                                        all_answers.append(f"**From {doc_name}:** {result_data}")
                            
                            # Display combined results
                            if all_answers:
                                st.markdown("### 📋 Combined Query Results")
                                st.markdown("#### 💡 Answers from all documents:")
                                for answer in all_answers:
                                    st.markdown(f"• {answer}")
                                
                                # Display images from all sources
                                image_sources = [s for s in all_sources if 'path' in s]
                                if image_sources:
                                    st.markdown("#### 🖼️ Visual References from all documents:")
                                    
                                    for source in image_sources:
                                        st.markdown(f"**From {source['from_document']}:**")
                                        display_image_from_path(source['path'], f"Reference from {source['from_document']}")
                                
                                # Display text sources
                                text_sources = [s for s in all_sources if 'path' not in s]
                                if text_sources:
                                    st.markdown("#### 📚 Text Sources:")
                                    for i, source in enumerate(text_sources, 1):
                                        with st.expander(f"Source {i} from {source.get('from_document', 'Unknown')}"):
                                            st.markdown(f"**Content:** {source.get('content', 'No content available')}")
                                            if 'score' in source:
                                                st.markdown(f"**Relevance Score:** {source['score']:.3f}")
                
                elif query_button and not query_text.strip():
                    st.warning("⚠️ Please enter a question to search for.")
            else:
                st.warning("⚠️ Please select at least one document to query.")
        else:
            st.info("📭 No documents available. Please upload documents first!")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "📚 Document Query System | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()