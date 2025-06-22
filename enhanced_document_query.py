
import logging
from typing import List,Dict,Any,Optional
import os
import base64
import json
from retriever import AdvancedRetriever
import numpy as np
from llm import GeminiEmbeddingModel
from processor import GeminiAnswerGenerator
import re
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from m2 import EnhancedMultimodalDocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class EnhancedMultimodalQuerySystem:
    """Enhanced query system with persistent storage integration"""
    
    def __init__(self, gemini_api_key: str, storage_base_path: str = "document_storage"):
        self.document_processor=EnhancedMultimodalDocumentProcessor(gemini_api_key=gemini_api_key)
        self.embedding_model = GeminiEmbeddingModel(gemini_api_key)
        self.answer_generator = GeminiAnswerGenerator(gemini_api_key)
        self.storage_base_path = Path(storage_base_path)
        self.metadata_file = self.storage_base_path / "documents_metadata.json"
        
        # Cache for loaded documents to avoid repeated file I/O
        self.loaded_documents = {}  # doc_id -> {'chunks': chunks, 'embeddings': embeddings, 'retriever': retriever}
        
        logger.info(f"Query system initialized with storage path: {self.storage_base_path}")
    
    def load_document_from_storage(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Load processed document from persistent storage"""
        try:
            # Check if already loaded in cache
            if doc_id in self.loaded_documents:
                logger.info(f"Loading document {doc_id} from cache")
                return self.loaded_documents[doc_id]
            
            doc_storage_path = self.storage_base_path / doc_id
            
            if not self.document_processor.document_exists_and_valid(doc_id):
                logger.error(f"Document {doc_id} not found or invalid")
                return None
            
            logger.info(f"Loading document {doc_id} from storage: {doc_storage_path}")
            
            # Load chunks
            chunks_file = doc_storage_path / "chunks.pkl"
            with open(chunks_file, 'rb') as f:
                chunks = pickle.load(f)
            
            # Load embeddings
            embeddings_file = doc_storage_path / "embeddings.pkl"
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Load subdirs info
            subdirs_file = doc_storage_path / "subdirs.json"
            with open(subdirs_file, 'r') as f:
                subdirs_data = json.load(f)
                subdirs = {k: Path(v) for k, v in subdirs_data.items()}
            
            # Create retriever
            retriever = AdvancedRetriever(chunks, embeddings)
            
            # Cache the loaded document
            document_data = {
                'chunks': chunks,
                'embeddings': embeddings,
                'subdirs': subdirs,
                'retriever': retriever
            }
            
            self.loaded_documents[doc_id] = document_data
            
            logger.info(f"Successfully loaded document {doc_id} with {len(chunks)} chunks")
            return document_data
        
        except Exception as e:
            logger.error(f"Error loading document {doc_id}: {e}")
            return None   
    
    def query(self, query_text: str, top_k: int = 5, 
              doc_ids: Optional[List[str]] = None,
              load_multimedia: bool = True) -> Dict[str, Any]:
        """Enhanced query with persistent storage support"""
        # Get available documents if no specific doc_ids provided
        if doc_ids is None:
            metadata = self.document_processor.load_metadata()
            doc_ids = list(metadata.keys())
            logger.info(f"No doc_ids specified, searching across all {len(doc_ids)} documents")
        
        if not doc_ids:
            return {
                "error": "No documents selected for querying",
                "available_documents": self.document_processor.list_processed_documents()
            }
        
        all_results = []
        loaded_doc_count = 0
        
        # Search across specified documents
        for doc_id in doc_ids:
            logger.info(f"Processing document: {doc_id}")
            
            # Load document from storage
            document_data = self.load_document_from_storage(doc_id)
            if document_data is None:
                logger.warning(f"Skipping document {doc_id} - could not load")
                continue
            
            loaded_doc_count += 1
            retriever = document_data['retriever']
            chunks = document_data['chunks']
            
            # Perform hybrid search
            try:
                doc_results = retriever.hybrid_search(query_text, self.embedding_model, top_k * 2)
                
                for chunk_idx, score in doc_results:
                    if chunk_idx < len(chunks):  # Safety check
                        chunk = chunks[chunk_idx]
                        all_results.append({
                            'chunk': chunk,
                            'score': score,
                            'doc_id': doc_id
                        })
                
                logger.info(f"Retrieved {len(doc_results)} results from document {doc_id}")
                
            except Exception as e:
                logger.error(f"Error searching in document {doc_id}: {e}")
                continue
        
        if not all_results:
            return {
                "error": f"No results found. Searched {loaded_doc_count} documents.",
                "query": query_text,
                "searched_documents": loaded_doc_count
            }
        
        # Sort all results by score and get top_k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_results = all_results[:top_k]
        
        logger.info(f"Found {len(all_results)} total results, returning top {len(top_results)}")
        
        # Prepare results for answer generation
        retrieved_chunks = []
        for result in top_results:
            chunk = result['chunk']
            retrieved_chunks.append({
                'content': chunk.content,
                'doc_id': result['doc_id'],
                'page_num': chunk.page_num,
                'chunk_id': chunk.chunk_id,
                'score': result['score'],
                'metadata': chunk.metadata
            })
        
        # Load multimedia content if requested
        multimedia_content = {'images': [], 'tables': []}
        if load_multimedia:
            try:
                multimedia_content = self.load_comprehensive_multimedia_content( query=query_text, results=top_results,similarity_threshold=0.3)
                logger.info(f"Loaded multimedia: {len(multimedia_content['images'])} images, {len(multimedia_content['tables'])} tables")
            except Exception as e:
                logger.error(f"Error loading multimedia content: {e}")
        
        # Generate comprehensive answer
        try:
            answer = self.answer_generator.generate_answer(
                query_text, retrieved_chunks, multimedia_content
            )
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "Sorry, I encountered an error while generating the answer, but I found relevant content in the documents."
        
        return {
            'query': query_text,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'multimedia_content': multimedia_content,
            'total_results': len(all_results),
            'top_results_count': len(top_results),
            'searched_documents': loaded_doc_count,
            'document_sources': list(set(result['doc_id'] for result in top_results))
        }
    
    def query_specific_document(self, doc_id: str, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Query a specific document by doc_id"""
        return self.query(query_text, top_k=top_k, doc_ids=[doc_id])
    
    def extract_ai_descriptions_from_chunk(self, chunk_text: str) -> List[Dict[str, str]]:
        """Extract AI descriptions and summaries from chunk text"""
        descriptions = []
        
        # Pattern for images with descriptions
        image_pattern = r'\[IMAGE_REF:([^|]+)\|IMAGE_CONTEXT:([^|]+)\|DESCRIPTION:([^\]]+)\]'
        image_matches = re.findall(image_pattern, chunk_text)
        
        for path, context, description in image_matches:
            descriptions.append({
                'type': 'image',
                'path': path,
                'context': context,
                'description': description.strip(),
                'searchable_text': f"{context} {description}".strip()
            })
        
        # Pattern for tables with summaries
        table_pattern = r'\[TABLE_REF:([^|]+)\|TABLE_CONTEXT:([^|]+)\|SUMMARY:([^\]]+)\]'
        table_matches = re.findall(table_pattern, chunk_text)
        
        for path, context, summary in table_matches:
            descriptions.append({
                'type': 'table',
                'path': path,
                'context': context,
                'description': summary.strip(),
                'searchable_text': f"{context} {summary}".strip()
            })
        
        return descriptions
    
    def calculate_semantic_similarity(self, query: str, descriptions: List[str]) -> List[float]:
        """Calculate semantic similarity between query and descriptions"""
        if not descriptions:
            return []
        
        try:
            # Encode query and descriptions
            query_embedding = self.embedding_model.embed_batch([query])
            desc_embeddings = self.embedding_model.embed_batch(descriptions)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, desc_embeddings)[0]
            return similarities.tolist()
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return [0.0] * len(descriptions)
    
    def filter_relevant_multimedia(self, query: str, all_descriptions: List[Dict], 
                                 similarity_threshold: float = 0.3) -> List[Dict]:
        """Filter multimedia content based on semantic similarity to query"""
        if not all_descriptions:
            return []
        
        # Extract searchable texts
        searchable_texts = [desc['searchable_text'] for desc in all_descriptions]
        
        # Calculate similarities
        similarities = self.calculate_semantic_similarity(query, searchable_texts)
        
        # Filter based on threshold
        relevant_items = []
        for i, (desc, similarity) in enumerate(zip(all_descriptions, similarities)):
            if similarity >= similarity_threshold:
                desc['similarity_score'] = similarity
                relevant_items.append(desc)
        
        # Sort by similarity score (highest first)
        relevant_items.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return relevant_items

    def load_comprehensive_multimedia_content(self, query: str, results: List[Dict], 
                                            similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """Load all relevant multimedia content from results with AI-based filtering"""
        multimedia = {
            'images': [],
            'tables': [],
            'ai_descriptions': []
        }
        
        seen_paths = set()
        all_descriptions = []
        
        # Phase 1: Extract all AI descriptions from chunks
        for result in results:
            chunk = result['chunk']
            
            # Extract AI descriptions from chunk text
            chunk_descriptions = self.extract_ai_descriptions_from_chunk(chunk.content)
            
            for desc in chunk_descriptions:
                desc.update({
                    'doc_id': result['doc_id'],
                    'page_num': chunk.page_num,
                    'chunk_id': chunk.chunk_id
                })
                all_descriptions.append(desc)
        
        # Phase 2: Filter descriptions based on query relevance
        relevant_descriptions = self.filter_relevant_multimedia(query, all_descriptions, similarity_threshold)
        
        # Phase 3: Load actual multimedia files for relevant items
        for desc in relevant_descriptions:
            path = desc['path']
            
            if path in seen_paths or not os.path.exists(path):
                continue
                
            seen_paths.add(path)
            
            try:
                if desc['type'] == 'image' and path.endswith(('.png', '.jpg', '.jpeg')):
                    # Load image
                    with open(path, 'rb') as f:

                        import base64
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Load metadata if available
                    metadata_path = path.replace('.png', '_metadata.json').replace('.jpg', '_metadata.json').replace('.jpeg', '_metadata.json')
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    multimedia['images'].append({
                        'path': path,
                        'data': img_data,
                        'metadata': metadata,
                        'doc_id': desc['doc_id'],
                        'page_num': desc['page_num'],
                        'chunk_id': desc['chunk_id'],
                        'ai_description': desc['description'],
                        'context': desc['context'],
                        'similarity_score': desc['similarity_score'],
                        'relevance_reason': f"Matched query with {desc['similarity_score']:.2f} similarity"
                    })
                
                elif desc['type'] == 'table' and path.endswith(('.txt', '.csv')):
                    # Load table content
                    with open(path, 'r', encoding='utf-8') as f:
                        table_content = f.read()
                    
                    # Load metadata if available
                    metadata_path = path.replace('.txt', '_metadata.json').replace('.csv', '_metadata.json')
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    multimedia['tables'].append({
                        'path': path,
                        'content': table_content,
                        'metadata': metadata,
                        'doc_id': desc['doc_id'],
                        'page_num': desc['page_num'],
                        'chunk_id': desc['chunk_id'],
                        'ai_summary': desc['description'],
                        'context': desc['context'],
                        'similarity_score': desc['similarity_score'],
                        'relevance_reason': f"Matched query with {desc['similarity_score']:.2f} similarity"
                    })
            
            except Exception as e:
                logger.error(f"Error loading multimedia content from {path}: {e}")
        
        # Store all AI descriptions for reference
        multimedia['ai_descriptions'] = relevant_descriptions
        
        logger.info(f"Loaded {len(multimedia['images'])} relevant images and {len(multimedia['tables'])} relevant tables")
        
        return multimedia

