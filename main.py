"""
FastAPI server for CSE Document Question Answer Chatbot
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import numpy as np
from typing import List, Dict, Optional
import logging
import tempfile
import shutil

from chatbot_core import DocumentProcessor, CSEChatbot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CSE Chatbot API",
    description="Document-based Question Answering System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chatbot instance
chatbot: Optional[CSEChatbot] = None
KNOWLEDGE_BASE_PATH = "knowledge_base.json"
UPLOAD_DIR = "./uploaded_documents"


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    query: str
    response: str
    relevant_documents: List[Dict]
    processing_time: float


class StatusResponse(BaseModel):
    status: str
    message: str
    documents_loaded: int


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup"""
    global chatbot
    logger.info("Initializing chatbot...")
    
    chatbot = CSEChatbot()
    
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Load existing knowledge base if available
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        logger.info("Loading existing knowledge base...")
        chatbot.load_knowledge_base(KNOWLEDGE_BASE_PATH)
        logger.info(f"Loaded {len(chatbot.documents)} document chunks from knowledge base")
    else:
        logger.info("No existing knowledge base found. Please upload documents.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down chatbot...")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "CSE Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "status": "/status",
            "query": "/query",
            "upload": "/upload",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None
    }


@app.get("/status", tags=["Status"], response_model=StatusResponse)
async def status():
    """Get chatbot status"""
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    documents_count = len(chatbot.documents) if chatbot.documents else 0
    
    return StatusResponse(
        status="ready" if documents_count > 0 else "no_documents",
        message=f"Knowledge base contains {documents_count} document chunks" if documents_count > 0 
                else "No documents loaded. Please upload documents first.",
        documents_loaded=documents_count
    )


@app.post("/upload", tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Upload documents (PDF, DOCX, TXT) and process them
    """
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create temporary directory for uploads
    temp_dir = tempfile.mkdtemp()
    uploaded_files = []
    
    try:
        # Save uploaded files
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                uploaded_files.append({
                    "filename": file.filename,
                    "size": len(content)
                })
        
        # Process documents in background
        background_tasks.add_task(
            process_and_save_documents,
            temp_dir,
            uploaded_files
        )
        
        return {
            "status": "processing",
            "message": f"Received {len(uploaded_files)} file(s). Processing in background...",
            "files": uploaded_files
        }
    
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


def process_and_save_documents(directory: str, uploaded_files: List[Dict]):
    """
    Process uploaded documents and save knowledge base
    This runs in the background
    """
    try:
        logger.info(f"Processing {len(uploaded_files)} documents...")
        
        # Process documents
        docs = DocumentProcessor.process_directory(directory)
        chatbot.documents.extend(docs)
        
        # Generate embeddings
        if chatbot.documents:
            texts = [doc["content"] for doc in chatbot.documents]
            chatbot.embeddings = chatbot.encoder.encode(texts, show_progress_bar=False)
        
        # Save knowledge base
        chatbot.save_knowledge_base(KNOWLEDGE_BASE_PATH)
        
        logger.info(f"Successfully processed and saved {len(docs)} document chunks")
    
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
    
    finally:
        # Clean up temporary directory
        if os.path.exists(directory):
            shutil.rmtree(directory)


@app.post("/query", tags=["Query"], response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Submit a query to the chatbot
    """
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if len(chatbot.documents) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = chatbot.get_response(request.query, top_k=request.top_k)
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/knowledge-base", tags=["Documents"])
async def delete_knowledge_base():
    """
    Clear the knowledge base
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        chatbot.documents = []
        chatbot.embeddings = None
        
        if os.path.exists(KNOWLEDGE_BASE_PATH):
            os.remove(KNOWLEDGE_BASE_PATH)
        
        return {
            "status": "success",
            "message": "Knowledge base cleared"
        }
    
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-knowledge-base", tags=["Documents"])
async def reload_knowledge_base():
    """
    Reload knowledge base from file
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise HTTPException(status_code=400, detail="Knowledge base file not found")
    
    try:
        chatbot.load_knowledge_base(KNOWLEDGE_BASE_PATH)
        return {
            "status": "success",
            "message": f"Knowledge base reloaded with {len(chatbot.documents)} documents"
        }
    
    except Exception as e:
        logger.error(f"Error reloading knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
