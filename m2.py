
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import pickle
import os 
# Core libraries
import pymupdf  # fitz
import tabula

from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

from llm import GeminiEmbeddingModel
from processor import GeminiAnswerGenerator
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Enhanced document chunk with additional metadata for advanced retrieval"""
    chunk_id: str
    page_num: int
    chunk_type: str  # 'text', 'image', 'table', 'page'
    content: str
    file_path: str
    metadata: Dict[str, Any]
    embedded_paths: List[str]  
    bbox: Optional[Tuple[float, float, float, float]] = None
    keywords: List[str] = None  
    semantic_density: float = 0.0  
    cross_references: List[str] = None  


class EnhancedMultimodalDocumentProcessor:
    """Enhanced document processor with page-wise chunking and persistent storage"""
    
    def __init__(self, base_output_dir: str = "./processed_docs", gemini_api_key: str = None, 
                 max_chunk_size: int = 2000, enable_smart_splitting: bool = True,
                 storage_base_path: str = "document_storage"):
        self.base_output_dir = Path(base_output_dir)
        self.storage_base_path = Path(storage_base_path)  # New: persistent storage path
        self.max_chunk_size = max_chunk_size
        self.enable_smart_splitting = enable_smart_splitting
        
        # Create storage directories
        self.storage_base_path.mkdir(exist_ok=True)
        self.metadata_file = self.storage_base_path / "documents_metadata.json"
        self.GeminiAnswerGenerator=GeminiAnswerGenerator(api_key=gemini_api_key)
        
        # Keep recursive splitter as fallback for very large pages
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size, 
            chunk_overlap=200, 
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")
            
        
        self.embedding_model = GeminiEmbeddingModel(gemini_api_key)
        self.supported_formats = {'.pdf', '.docx', '.txt', '.pptx', '.xlsx', '.csv'}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load document metadata from storage"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save document metadata to storage"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def get_file_hash(self, filepath: str) -> str:
        """Generate hash for file to check if already processed"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_document_processed(self, file_hash: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Check if document with given hash is already processed"""
        for doc_id, doc_info in metadata.items():
            if doc_info.get('file_hash') == file_hash:
                return doc_id
        return None
    
    def document_exists_and_valid(self, doc_id: str) -> bool:
        """Check if processed document files exist and are valid"""
        try:
            doc_storage_path = self.storage_base_path / doc_id
            if not doc_storage_path.exists():
                return False
            
            # Check for required files
            required_files = [
                doc_storage_path / "chunks.pkl",
                doc_storage_path / "embeddings.pkl",
                doc_storage_path / "subdirs.json"
            ]
            
            return all(f.exists() for f in required_files)
        except Exception:
            return False
    
    def process_document(self, filepath: str, custom_name: Optional[str] = None, 
                        force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Enhanced document processing with duplicate detection and persistent storage
        
        Args:
            filepath: Path to document file
            custom_name: Optional custom name for the document
            force_reprocess: Force reprocessing even if document exists
        """
        file_ext = Path(filepath).suffix.lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load existing metadata
        metadata = self.load_metadata()
        
        # Check if document already processed (unless forced)
        if not force_reprocess:
            file_hash = self.get_file_hash(filepath)
            existing_doc_id = self.is_document_processed(file_hash, metadata)
            
            if existing_doc_id and self.document_exists_and_valid(existing_doc_id):
                logger.info(f"Document already processed: {existing_doc_id}")
                
                # Load existing processed data
                doc_storage_path = self.storage_base_path / existing_doc_id
                chunks, embeddings, subdirs = self.load_processed_document(doc_storage_path)
                
                return {
                    'status': 'already_processed',
                    'doc_id': existing_doc_id,
                    # 'chunks': chunks,
                    # 'embeddings': embeddings,
                    'subdirs': str(subdirs),
                    'processing_stats': metadata[existing_doc_id]['processing_stats'],
                    'message': 'Document already exists and was loaded from storage'
                }
        
        # Generate new doc_id and process
        doc_id = self.generate_doc_id(filepath)
        doc_storage_path = self.storage_base_path / doc_id
        doc_storage_path.mkdir(exist_ok=True)
        
        # Create subdirectories in storage path
        subdirs = self.create_directory_structure_in_storage(doc_id)
        
        logger.info(f"Processing document: {filepath}")
        logger.info(f"Document ID: {doc_id}")
        logger.info(f"Using page-wise chunking with max_chunk_size: {self.max_chunk_size}")
        

        if file_ext == '.pdf':
            chunks = self.process_pdf_with_ai_enhanced_chunking(filepath, doc_id, subdirs)
        # elif file_ext == '.docx':
        #     chunks = self.process_docx_with_page_wise_chunking(filepath, doc_id, subdirs)
        # elif file_ext == '.txt':
        #     chunks = self.process_txt_with_chunking(filepath, doc_id, subdirs)
        # elif file_ext == '.pptx':
        #     chunks = self.process_pptx_with_slide_wise_chunking(filepath, doc_id, subdirs)
        # elif file_ext in ['.xlsx', '.csv']:
        #     chunks = self.process_spreadsheet_with_chunking(filepath, doc_id, subdirs)
        # else:
        #     raise ValueError(f"Processing for {file_ext} not yet implemented")
        
        if not chunks:
            raise ValueError("No content could be extracted from the document")
        

        
        # Create embeddings
        logger.info("Generating embeddings...")
        embeddings = self.create_embeddings(chunks)
                # Calculate page statistics for reporting
        page_stats = {}
        for chunk in chunks:
            page_num = chunk.page_num
            if page_num not in page_stats:
                page_stats[page_num] = {
                    'chunk_count': 0,
                    'has_images': False,
                    'has_tables': False,
                    'total_length': 0
                }
            page_stats[page_num]['chunk_count'] += 1
            page_stats[page_num]['total_length'] += len(chunk.content)
            if chunk.metadata.get('page_has_images'):
                page_stats[page_num]['has_images'] = True
            if chunk.metadata.get('page_has_tables'):
                page_stats[page_num]['has_tables'] = True
        
        # Save processed data to storage
        self.save_processed_data_to_storage(doc_id, chunks, embeddings, subdirs, filepath)
        
        # Update metadata
        file_path = Path(filepath)
        doc_name = custom_name or file_path.stem
        file_hash = self.get_file_hash(filepath)
        
        processing_stats = {
            'total_chunks': len(chunks),
            'total_pages': len(set(chunk.page_num for chunk in chunks)),
            'avg_chunks_per_page': len(chunks) / len(set(chunk.page_num for chunk in chunks)) if chunks else 0,
            'chunks_with_images': sum(1 for chunk in chunks if chunk.metadata.get('page_has_images', False)),
            'chunks_with_tables': sum(1 for chunk in chunks if chunk.metadata.get('page_has_tables', False)),
            'total_characters': sum(len(chunk.content) for chunk in chunks),
            'avg_chunk_size': sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0
        }
        
        metadata[doc_id] = {
            'doc_id': doc_id,
            'doc_name': doc_name,
            'original_filepath': str(file_path.absolute()),
            'file_hash': file_hash,
            'file_size': file_path.stat().st_size,
            'processed_date': datetime.now().isoformat(),
            'processing_stats': processing_stats,
            'storage_path': str(doc_storage_path),
            'processing_method': 'page_wise_chunking',
            'chunking_config': {
                'max_chunk_size': self.max_chunk_size,
                'smart_splitting_enabled': self.enable_smart_splitting
            }
        }
        
        self.save_metadata(metadata)
        
        logger.info(f"Processing complete! Document ID: {doc_id}")
        logger.info(f"Created {len(chunks)} chunks from {processing_stats['total_pages']} pages")
        
        return {
            'status': 'success',
            'doc_id': doc_id,
            # 'chunks': chunks,
            # 'embeddings': embeddings,
            'subdirs': str(subdirs),
            'processing_stats': processing_stats,
            'message': 'Document processed and stored successfully'
        }
    
    def create_directory_structure_in_storage(self, doc_id: str) -> Dict[str, Path]:
        """Create organized directory structure in storage path"""
        doc_dir = self.storage_base_path / doc_id
        subdirs = {
            'images': doc_dir / 'images',
            'tables': doc_dir / 'tables', 
            'text_chunks': doc_dir / 'text_chunks',
            # 'metadata': doc_dir / 'metadata'
        }
        
        for subdir in subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
            
        return subdirs
 
    def save_processed_data_to_storage(self, doc_id: str, chunks: List[DocumentChunk], 
                                     embeddings: np.ndarray, subdirs: Dict[str, Path], filepath: str):
        """Save all processed data to persistent storage"""
        doc_storage_path = self.storage_base_path / doc_id
        
        # Save chunks
        chunks_file = doc_storage_path / "chunks.pkl"
        with open(chunks_file, 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save embeddings
        embeddings_file = doc_storage_path / "embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save subdirs info
        subdirs_file = doc_storage_path / "subdirs.json"
        with open(subdirs_file, 'w') as f:
            json.dump({k: str(v) for k, v in subdirs.items()}, f)

        logger.info(f"Processed data saved to storage: {doc_storage_path}")
    
    def load_processed_document(self, doc_storage_path: Path) -> tuple:
        """Load processed document from storage"""
        try:
            # Load chunks
            chunks_file = doc_storage_path / "chunks.pkl"
            with open(chunks_file, 'rb') as f:
                chunks = pickle.load(f)
            
            # Load embeddings
            embeddings_file = doc_storage_path / "embeddings.pkl"
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Load subdirs
            subdirs_file = doc_storage_path / "subdirs.json"
            with open(subdirs_file, 'r') as f:
                subdirs_data = json.load(f)
                subdirs = {k: Path(v) for k, v in subdirs_data.items()}
            
            return chunks, embeddings, subdirs
        
        except Exception as e:
            logger.error(f"Error loading processed document from {doc_storage_path}: {e}")
            raise
    
    def list_processed_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents"""
        metadata = self.load_metadata()
        return [
            {
                'doc_id': doc_id,
                'doc_name': info['doc_name'],
                'processed_date': info['processed_date'],
                'processing_stats': info['processing_stats'],
                'file_size': info['file_size']
            }
            for doc_id, info in metadata.items()
            if self.document_exists_and_valid(doc_id)
        ]
    

    def delete_processed_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a processed document and its files"""
        try:
            metadata = self.load_metadata()
            
            if doc_id not in metadata:
                return {'status': 'error', 'message': 'Document not found'}
            
            # Remove storage directory
            doc_storage_path = self.storage_base_path / doc_id
            if doc_storage_path.exists():
                import shutil
                shutil.rmtree(doc_storage_path)
            
            # Remove from metadata
            doc_name = metadata[doc_id]['doc_name']
            del metadata[doc_id]
            self.save_metadata(metadata)
            
            logger.info(f"Deleted document: {doc_name} ({doc_id})")
            return {
                'status': 'success',
                'message': f'Document "{doc_name}" deleted successfully'
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return {'status': 'error', 'message': str(e)}
 
    def generate_doc_id(self, filepath: str) -> str:
        """Generate unique document ID with better collision avoidance"""
        filename = Path(filepath).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include microseconds
        
        # Create hash from file path + current time for uniqueness
        hash_input = f"{filepath}_{timestamp}".encode()
        hash_obj = hashlib.md5(hash_input)
        
        return f"{filename}_{hash_obj.hexdigest()[:8]}_{timestamp}"
    
    def smart_page_chunking(self, page_text: str, page_num: int) -> List[str]:
        """Smart page-wise chunking that preserves content integrity"""
        # If page is small enough, keep as single chunk
        if len(page_text) <= self.max_chunk_size:
            return [page_text]
        
        if not self.enable_smart_splitting:
            # Use fallback splitter for large pages
            return self.fallback_splitter.split_text(page_text)
        
        chunks = []
        
        # Try to split by logical sections first
        sections = self.split_by_logical_sections(page_text)
        
        current_chunk = ""
        for section in sections:
            # If adding this section would exceed max size
            if len(current_chunk + section) > self.max_chunk_size:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If section itself is too large, split it further
                if len(section) > self.max_chunk_size:
                    section_chunks = self.fallback_splitter.split_text(section)
                    chunks.extend(section_chunks[:-1])  # Add all but last
                    current_chunk = section_chunks[-1] if section_chunks else ""
                else:
                    current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [page_text]
    
    def split_by_logical_sections(self, text: str) -> List[str]:
        """Split text by logical sections (headings, paragraphs, etc.)"""
        sections = []
        
        # Split by double newlines first (paragraphs)
        paragraphs = text.split('\n\n')
        
        current_section = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this looks like a heading (simple heuristic)
            is_heading = (
                len(paragraph.split('\n')[0]) < 100 and  # Short first line
                (paragraph.isupper() or  # All caps
                 paragraph.split('\n')[0].count(' ') < 8)  # Few words
            )
            
            if is_heading and current_section:
                # Start new section
                sections.append(current_section)
                current_section = paragraph
            else:
                current_section += "\n\n" + paragraph if current_section else paragraph
        
        if current_section:
            sections.append(current_section)
        
        return sections if sections else [text]
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)  # Capitalized words
        words.extend(re.findall(r'\b\w{6,}\b', text.lower()))  # Long words
        return list(set(words))[:10]  # Top 10 unique keywords
    
    def calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic importance of text chunk"""
        factors = {
            'length': min(len(text) / 1000, 1.0),
            'has_numbers': 0.1 if re.search(r'\d+', text) else 0,
            'has_capitals': len(re.findall(r'\b[A-Z][a-zA-Z]+\b', text)) / 100,
            'sentence_count': min(len(text.split('.')) / 10, 1.0)
        }
        return sum(factors.values()) / len(factors)
        

    def extract_paths_from_chunk(self, chunk_text: str) -> List[str]:
        """Enhanced path extraction with context"""
        pattern = r'\[(IMAGE_REF|TABLE_REF):([^|]+)\|[^\]]*\]'
        matches = re.findall(pattern, chunk_text)
        return [match[1] for match in matches]
    def create_page_summary(self, page_text: str, page_num: int, 
                           image_paths: List[str], table_paths: List[str]) -> str:
        """Create a summary of the page content"""
        summary_parts = [f"Page {page_num + 1} Summary:"]
        
        # Text summary
        if page_text.strip():
            # You can add AI summarization here if needed
            text_preview = page_text[:200] + "..." if len(page_text) > 200 else page_text
            summary_parts.append(f"Text Content: {text_preview}")
        else:
            summary_parts.append("Text Content: None (Image-based page)")
        
        # Visual elements
        if image_paths:
            summary_parts.append(f"- Images: {len(image_paths)} found")
        if table_paths:
            summary_parts.append(f"- Tables: {len(table_paths)} found")
        
        # Key topics (simple extraction)
        keywords = self.extract_keywords(page_text)
        if keywords:
            summary_parts.append(f"- Key terms: {', '.join(keywords[:5])}")
        
        return "\n".join(summary_parts)
    
    def create_enhanced_text_with_paths_and_summaries(self, text: str, image_paths: List[str], 
    table_paths: List[str], page_num: int) -> str:
        """Enhanced text creation with AI-generated summaries"""
        enhanced_text = text
        
        # Add context-aware references with AI descriptions
        if image_paths:
            image_refs = []
            for i, path in enumerate(image_paths):
                # Get AI description
                image_description = self.GeminiAnswerGenerator.analyze_image_with_gemini(path)
                
                image_refs.append(
                    f"[IMAGE_REF:{path}|IMAGE_CONTEXT:Page {page_num} Image {i+1}|"
                    f"\n DESCRIPTION:{image_description}]"
                )
            enhanced_text += f"\n\n--- Visual Content ---\n" + "\n".join(image_refs)
        
        if table_paths:
            table_refs = []
            for i, path in enumerate(table_paths):
                # Get AI summary
                table_summary = self.GeminiAnswerGenerator.analyze_table_with_gemini(path)
                
                
                
                table_refs.append(
                    f"[TABLE_REF:{path}|TABLE_CONTEXT:Page {page_num} Table {i+1}|"
                    f"SUMMARY:{table_summary}]"
                )
            enhanced_text += f"\n\n--- Tabular Data ---\n" + "\n".join(table_refs)
        
        return enhanced_text
    
    def extract_images_from_page_with_analysis(self, page, page_num: int, doc_id: str, 
                                             subdirs: Dict[str, Path]) -> List[str]:
        """Enhanced image extraction with AI analysis"""
        image_paths = []
        images = page.get_images()
        
        for idx, img in enumerate(images):
            try:
                xref = img[0]
                pix = pymupdf.Pixmap(page.parent, xref)
                
                if pix.width < 50 or pix.height < 50:
                    continue
                
                image_filename = f"{doc_id}_page{page_num}_img{idx}_{xref}.png"
                image_path = subdirs['images'] / image_filename
                
                pix.save(str(image_path))
                
                # Save enhanced metadata with AI description
                metadata = {
                    'width': pix.width,
                    'height': pix.height,
                    'page_num': page_num,
                    'image_index': idx,
                    'file_path': str(image_path),
                    # 'ai_description': ai_description,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                metadata_file = subdirs['images'] / f"{image_filename}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                image_paths.append(str(image_path))
                pix = None
                
            except Exception as e:
                logger.error(f"Error extracting image {idx} from page {page_num}: {e}")
        
        return image_paths
    def extract_tables_from_page_with_analysis(self, doc, page_num: int, doc_id: str, 
                                             subdirs: Dict[str, Path]) -> List[str]:
        """Enhanced table extraction with AI analysis"""
        table_paths = []
        
        try:
            tables = tabula.read_pdf(doc.name, pages=page_num + 1, multiple_tables=True, silent=True)
            
            for table_idx, table in enumerate(tables):
                if table.empty:
                    continue
                    
                table_filename = f"{doc_id}_page{page_num}_table{table_idx}"
                
                # Save as CSV
                csv_path = subdirs['tables'] / f"{table_filename}.csv"
                table.to_csv(csv_path, index=False)
                
                # Save as formatted text
                text_path = subdirs['tables'] / f"{table_filename}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(table.to_string(index=False))
          
                # Save enhanced metadata with AI summary
                metadata = {
                    'shape': table.shape,
                    'columns': list(table.columns),
                    'page_num': page_num,
                    'table_index': table_idx,
                    # 'ai_summary': ai_summary,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                metadata_file = subdirs['tables'] / f"{table_filename}_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                
                table_paths.append(str(text_path))
                
        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {e}")
        
        return table_paths
    
    def process_pdf_with_ai_enhanced_chunking(self, filepath: str, doc_id: str, 
                                            subdirs: Dict[str, Path]) -> List[DocumentChunk]:
        """Process PDF with AI-enhanced content analysis"""
        doc = pymupdf.open(filepath)
        chunks = []
        
        logger.info(f"Processing {len(doc)} pages with AI-enhanced analysis...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}/{len(doc)} with AI analysis")
            
            # Extract images and tables with AI analysis
            image_paths = self.extract_images_from_page_with_analysis(page, page_num, doc_id, subdirs)
            table_paths = self.extract_tables_from_page_with_analysis(doc, page_num, doc_id, subdirs)
            
            # Get full page text
            full_text = page.get_text()
            
            # Skip empty pages
            if not full_text.strip() and not image_paths and not table_paths:
                logger.info(f"Skipping empty page {page_num + 1}")
                continue
            base_text = full_text if full_text.strip() else ""
            enhanced_text = self.create_enhanced_text_with_paths_and_summaries(
                base_text, image_paths, table_paths, page_num
            )
             # For pages with no text content, create a meaningful base text from AI analysis
            if not base_text.strip() and (image_paths or table_paths):
                base_text = f"[PAGE {page_num + 1}] This page contains visual/tabular content with no text."
                enhanced_text = base_text + "\n" + enhanced_text
            
            # Create page summary for context
            page_summary = self.create_page_summary(base_text, page_num, image_paths, table_paths)
            
            if enhanced_text.strip():
                page_chunks = self.smart_page_chunking(enhanced_text, page_num)
            else:
                # Fallback for edge cases
                page_chunks = [f"Page {page_num + 1} content"]
            
            logger.info(f"Page {page_num + 1} split into {len(page_chunks)} chunks with AI enhancements")
            
            
            for chunk_idx, chunk_text in enumerate(page_chunks):
                # Add page summary to first chunk of each page
                if chunk_idx == 0:
                    chunk_text = page_summary + "\n\n" + chunk_text
                
                embedded_paths = self.extract_paths_from_chunk(chunk_text)
                keywords = self.extract_keywords(chunk_text)
                semantic_density = self.calculate_semantic_density(chunk_text)
                
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
                chunk_file = subdirs['text_chunks'] / f"{chunk_id}.txt"
                
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(chunk_text)
                
                # Enhanced DocumentChunk with AI analysis
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    page_num=page_num,
                    chunk_type='image_based' if not full_text.strip() and image_paths else 'text',
                    content=chunk_text,
                    file_path=str(chunk_file),
                    metadata={
                        'source_file': filepath,
                        'doc_id': doc_id,
                        'chunk_index': chunk_idx,
                        'page_num': page_num,
                        'is_page_start': chunk_idx == 0,
                        'total_page_chunks': len(page_chunks),
                        'page_has_images': len(image_paths) > 0,
                        'is_image_only_page': not full_text.strip() and len(image_paths) > 0,
                        'page_has_tables': len(table_paths) > 0,
                        'ai_enhanced': True,
                        'processing_timestamp': datetime.now().isoformat()
                    },
                    embedded_paths=embedded_paths,
                    keywords=keywords,
                    semantic_density=semantic_density,
                    cross_references=[]
                )
                chunks.append(chunk)
        
        doc.close()
        logger.info(f"Processed document into {len(chunks)} AI-enhanced chunks")
        return chunks

    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings using Gemini"""
        texts = [chunk.content for chunk in chunks]
        return self.embedding_model.embed_batch(texts)
