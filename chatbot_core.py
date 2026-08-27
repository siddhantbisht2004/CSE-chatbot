"""
Core chatbot logic extracted from the Jupyter notebook
Handles document processing and semantic search
"""
import os
import json
import numpy as np
from typing import List, Dict
import PyPDF2
import docx
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
        return text

    @staticmethod
    def read_word_file(file_path: str) -> str:
        """Read content from Word documents."""
        try:
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error reading Word document {file_path}: {str(e)}")
            return ""

    @staticmethod
    def process_directory(directory_path: str) -> List[Dict]:
        """Process all supported documents in a directory."""
        processed_docs = []
        supported_extensions = {'.txt', '.pdf', '.docx'}

        if not os.path.exists(directory_path):
            logger.warning(f"Directory {directory_path} does not exist")
            return processed_docs

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                extension = os.path.splitext(file)[1].lower()

                if extension not in supported_extensions:
                    continue

                try:
                    logger.info(f"Processing file: {file_path}")
                    
                    if extension == '.txt':
                        content = DocumentProcessor.read_text_file(file_path)
                    elif extension == '.pdf':
                        content = DocumentProcessor.read_pdf_file(file_path)
                    elif extension == '.docx':
                        content = DocumentProcessor.read_word_file(file_path)
                    else:
                        continue

                    if not content:
                        logger.warning(f"No content extracted from {file_path}")
                        continue

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

                    logger.info(f"Successfully processed {file_path} into {len(chunks)} chunks")

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
    """Main chatbot class for semantic search and document Q&A"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the chatbot with necessary components."""
        logger.info(f"Initializing chatbot with model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def load_documents(self, directory_path: str):
        """Load and process documents from the specified directory."""
        logger.info(f"Processing documents from {directory_path}")

        # Process documents
        docs = DocumentProcessor.process_directory(directory_path)
        self.documents = docs

        logger.info(f"Number of documents processed: {len(self.documents)}")
        
        if len(self.documents) > 0:
            logger.info(f"Sample document content: {self.documents[0]['content'][:100]}...")
        else:
            logger.warning("No documents were processed")
            return

        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [doc["content"] for doc in self.documents]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

        logger.info(f"Processed {len(self.documents)} document chunks")
        logger.info(f"Generated embeddings for {len(self.embeddings)} documents")

    def save_knowledge_base(self, file_path: str):
        """Save the processed documents and embeddings."""
        logger.info(f"Saving knowledge base to {file_path}")
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        logger.info(f"Knowledge base saved to {file_path}")

    def load_knowledge_base(self, file_path: str):
        """Load previously processed documents and embeddings."""
        logger.info(f"Loading knowledge base from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "documents" in data and "embeddings" in data:
                self.documents = data["documents"]
                self.embeddings = np.array(data["embeddings"]) if data["embeddings"] else None
                logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
            else:
                logger.error("JSON file does not contain required keys 'documents' and 'embeddings'")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")

    def get_response(self, query: str, top_k: int = 3) -> Dict:
        """Process query and return response with relevant context."""
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.encoder.encode(query)

        # Calculate similarities
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings found")
            return {
                "query": query,
                "response": "No information available",
                "relevant_documents": [],
                "processing_time": time.time() - start_time
            }

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Check if the highest similarity score is less than the threshold
        if similarities.max() < 0.2:
            logger.info(f"Query similarity below threshold (max: {similarities.max():.2f})")
            return {
                "query": query,
                "response": "No information available",
                "relevant_documents": [],
                "processing_time": time.time() - start_time
            }

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_docs = []
        for idx in top_indices:
            if idx < len(self.documents):
                doc = self.documents[idx]
                relevant_docs.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": float(similarities[idx])
                })
            else:
                logger.error(f"Index {idx} is out of range for documents list")

        # Generate response
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
            return "No information available"

        # Return the most relevant document's content
        return relevant_docs[0]["content"]
