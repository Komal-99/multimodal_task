import logging 
import google.generativeai as genai
from typing import List, Dict
from PIL import Image
import pandas as pd
import json
from flask import jsonify
import re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiAnswerGenerator:
    """Generate comprehensive answers using Gemini 2.5"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"): #gemini-2.0-flash-exp"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def generate_answer(self, query: str, retrieved_chunks: List[Dict], 
                       multimedia_content: Dict) -> str:
        """Generate comprehensive answer with AI-enhanced multimedia integration"""
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[Context {i+1}]:\n{chunk['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Prepare enhanced multimedia context with AI descriptions
        multimedia_context = ""
        
        if multimedia_content['images']:
            multimedia_context += f"\nRelevant Images Found ({len(multimedia_content['images'])}):\n"
            for i, img in enumerate(multimedia_content['images']):
                multimedia_context += f"Image {i+1}: {img['ai_description']} "
                multimedia_context += f"(Relevance: {img['similarity_score']:.2f}, {img['context']})\n"
        
        if multimedia_content['tables']:
            multimedia_context += f"\nRelevant Tables Found ({len(multimedia_content['tables'])}):\n"
            for i, table in enumerate(multimedia_content['tables']):
                multimedia_context += f"Table {i+1}: {table['ai_summary']} "
                multimedia_context += f"(Relevance: {table['similarity_score']:.2f}, {table['context']})\n"
        
        prompt = f"""
        Based on the following context from documents, provide a comprehensive answer to the user's query.
        
        Query: {query}
        
        Document Context:
        {context}
        
        Relevant Multimedia Content:
        {multimedia_context}
        
        Instructions:
        1. Provide a detailed, accurate answer based on the context
        2.For images and tables in multimedia context analyze use their description to frame a good answer.
        3. Use the AI-generated descriptions to explain what visual content shows
        4. If multimedia content contradicts or supplements text, note this explicitly
        5. If the answer is not fully contained in the context, state what information is available
        6. Maintain professional tone and structure
        7. Return images paths in paths feild.

         **Return only a valid JSON list of one object with the following format:**
        - "Answer": your generated answer
        - "paths": list of image/table file paths used

        ### Example Response Format (Do not include backticks, markdown, or formatting):
        [
        
            "Answer": "Your generated answer goes here...",
            "paths": ["path1", "path2"]
        
        ]
        """
        
        try:
            response = self.model.generate_content(prompt)
            print(response.text)
            cleaned = re.sub(r"```(?:json)?\n(.*?)```", r"\1", response.text, flags=re.DOTALL).strip()
            parsed = json.loads(cleaned)
            return parsed

        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            return "Sorry, I encountered an error while generating the answer."
    
    def get_multimedia_summary(self, multimedia_content: Dict) -> str:
        """Generate a summary of loaded multimedia content"""
        summary_parts = []
        
        if multimedia_content['images']:
            summary_parts.append(f"📸 {len(multimedia_content['images'])} relevant images:")
            for i, img in enumerate(multimedia_content['images'][:3]):  # Show top 3
                summary_parts.append(f"  • {img['ai_description']} (Score: {img['similarity_score']:.2f})")
            if len(multimedia_content['images']) > 3:
                summary_parts.append(f"  • ... and {len(multimedia_content['images']) - 3} more")
        
        if multimedia_content['tables']:
            summary_parts.append(f"📊 {len(multimedia_content['tables'])} relevant tables:")
            for i, table in enumerate(multimedia_content['tables'][:3]):  # Show top 3
                summary_parts.append(f"  • {table['ai_summary']} (Score: {table['similarity_score']:.2f})")
            if len(multimedia_content['tables']) > 3:
                summary_parts.append(f"  • ... and {len(multimedia_content['tables']) - 3} more")
        
        return "\n".join(summary_parts) if summary_parts else "No relevant multimedia content found."

    def analyze_image_with_gemini(self, image_path: str) -> str:
        """Generate detailed description of image using Gemini Vision"""
        try:
            # Load and prepare image
            image = Image.open(image_path)
            
            # Create prompt for image analysis
            prompt = """
            Analyze this image and provide a detailed but concise description focusing on:
            1. What the image depicts (objects, people, scenes, diagrams, charts, etc.)
            2. Key visual elements and their relationships
            3. Any text or data visible in the image
            4. Context that would be helpful for document understanding
            
            Keep the description informative in case of complete text image add whole text of image.
            """
            
            # Generate description using Gemini
            response = self.model.generate_content([prompt, image])
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            return "Image analysis unavailable"
    
    def analyze_table_with_gemini(self, table_path: str) -> str:
        """Generate summary of table content using Gemini"""
        try:
            # Read the table file
            if table_path.endswith('.csv'):
                df = pd.read_csv(table_path)
            else:
                # Read text file
                with open(table_path, 'r', encoding='utf-8') as f:
                    table_content = f.read()
                return self.summarize_table_text(table_content)
            
            # Convert first few rows to string for analysis
            table_preview = df.head(10).to_string(index=False)
            
            prompt = f"""
            Analyze this table data and provide a concise summary focusing on:
            1. What type of data this table contains
            2. Key columns and their meaning
            3. Notable patterns, trends, or important values
            4. The overall purpose/context of this data
            
            Table preview:
            {table_preview}
            
            Table shape: {df.shape[0]} rows, {df.shape[1]} columns
            
            Provide a brief summary (2-3 sentences max):
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing table with Gemini: {e}")
            return "Table analysis unavailable"
    
    def summarize_table_text(self, table_text: str) -> str:
        """Fallback method for text-based table analysis"""
        try:
            prompt = f"""
            Analyze this table data and provide a concise summary:
            
            {table_text[:1000]}...
            
            Focus on what type of data this is and its key characteristics (2-3 sentences max):
            """
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error in text table analysis: {e}")
            return "Table summary unavailable"

