import os
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import logging
from pathlib import Path
import json
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and text extraction from various file formats."""
    
    @staticmethod
    def read_text_file(file_path: str) -> str:
        """Read content from text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    @staticmethod
    def read_pdf_file(file_path: str) -> str:
        """Read content from PDF files."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def read_word_file(file_path: str) -> str:
        """Read content from Word documents."""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    @staticmethod
    def process_directory(directory_path: str) -> List[Dict]:
        """Process all supported documents in a directory."""
        processed_docs = []
        supported_extensions = {'.txt', '.pdf', '.docx'}
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                extension = os.path.splitext(file)[1].lower()
                
                if extension not in supported_extensions:
                    continue
                    
                try:
                    if extension == '.txt':
                        content = DocumentProcessor.read_text_file(file_path)
                    elif extension == '.pdf':
                        content = DocumentProcessor.read_pdf_file(file_path)
                    elif extension == '.docx':
                        content = DocumentProcessor.read_word_file(file_path)
                        
                    # Split content into manageable chunks
                    chunks = DocumentProcessor.chunk_text(content)
                    
                    for chunk in chunks:
                        processed_docs.append({
                            "content": chunk,
                            "metadata": {
                                "source": file_path,
                                "type": extension[1:],
                                "chunk_size": len(chunk)
                            }
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    
        return processed_docs
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Adjust chunk end to not split words
            if end < text_length:
                # Find the last space before chunk_size
                while end > start and text[end] != ' ':
                    end -= 1
                    
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                
            start = end - overlap
            
        return chunks

class CSEChatbot:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the chatbot with necessary components."""
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        
    def load_documents(self, directory_path: str):
        """Load and process documents from the specified directory."""
        logger.info(f"Processing documents from {directory_path}")
        
        # Process documents
        docs = DocumentProcessor.process_directory(directory_path)
        self.documents = docs
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [doc["content"] for doc in self.documents]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        logger.info(f"Processed {len(self.documents)} document chunks")
        
    def save_knowledge_base(self, file_path: str):
        """Save the processed documents and embeddings."""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
            
    def load_knowledge_base(self, file_path: str):
        """Load previously processed documents and embeddings."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.documents = data["documents"]
        self.embeddings = np.array(data["embeddings"]) if data["embeddings"] else None
        
    def get_response(self, query: str, top_k: int = 3) -> Dict:
        """Process query and return response with relevant context."""
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.encoder.encode(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            doc = self.documents[idx]
            relevant_docs.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "similarity": similarities[idx]
            })
        
        # Generate response (in a real system, you might want to use an LLM here)
        response = self._generate_simple_response(query, relevant_docs)
        
        processing_time = time.time() - start_time
        
        return {
            "query": query,
            "response": response,
            "relevant_documents": relevant_docs,
            "processing_time": processing_time
        }
    
    def _generate_simple_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate a simple response based on the most relevant document."""
        if not relevant_docs:
            return "I don't have enough information to answer that question."
            
        # Return the most relevant document's content
        return relevant_docs[0]["content"]

def main():
    # Example usage
    chatbot = CSEChatbot()
    
    # Check if saved knowledge base exists
    kb_path = "knowledge_base.json"
    if os.path.exists(kb_path):
        logger.info("Loading existing knowledge base...")
        chatbot.load_knowledge_base(kb_path)
    else:
        logger.info("Processing documents...")
        # Replace with your documents directory path
        chatbot.load_documents("./documents")
        chatbot.save_knowledge_base(kb_path)
    
    # Interactive query loop
    print("\nCSE Department Chatbot")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'quit':
            break
            
        if not query:
            continue
            
        try:
            result = chatbot.get_response(query)
            
            print("\nResponse:", result["response"])
            print("\nSource documents:")
            for i, doc in enumerate(result["relevant_documents"], 1):
                print(f"\n{i}. Similarity: {doc['similarity']:.2f}")
                print(f"Source: {doc['metadata']['source']}")
            print(f"\nProcessing time: {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print("Sorry, I encountered an error processing your query.")

if __name__ == "__main__":
    main()
