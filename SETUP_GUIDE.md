# CSE Chatbot - FastAPI Setup Guide

## 📋 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

## 🛠️ Installation Steps

### Step 1: Clone the Repository (if not already done)
```bash
git clone https://github.com/siddhantbisht2004/CSE-chatbot.git
cd CSE-chatbot
```

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first time you run the application, it will download the sentence-transformers model (~500 MB). This is a one-time download.

### Step 4: Run the Application

**Start the server:**
```bash
python main.py
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

## 🌐 Access the Application

### Option 1: Web Browser (Recommended for Testing)
1. Open your browser and go to: **http://localhost:8000/docs**
2. You'll see the **Swagger UI** - an interactive API documentation
3. Here you can test all endpoints directly

### Option 2: Using cURL Commands

**Check if server is running:**
```bash
curl http://localhost:8000/health
```

**Upload Documents:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@path/to/your/document.pdf"
```

**Query the Chatbot:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is CSE?",
    "top_k": 3
  }'
```

**Check Chatbot Status:**
```bash
curl http://localhost:8000/status
```

### Option 3: Using Python Requests
```python
import requests

# Query the chatbot
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What courses are available?",
        "top_k": 3
    }
)
print(response.json())
```

## 📁 Project Structure
```
CSE-chatbot/
├── main.py                          # FastAPI server
├── chatbot_core.py                  # Core chatbot logic
├── requirements.txt                 # Python dependencies
├── knowledge_base.json              # Saved embeddings (auto-created)
├── uploaded_files/                  # Directory for uploaded documents
│   └── (your PDFs, DOCX, TXT files)
└── README.md
```

## 🚀 Complete Example Workflow

### 1. Start Server
```bash
python main.py
```

### 2. Open Browser
Navigate to: **http://localhost:8000/docs**

### 3. Upload Documents
- Click the **"POST /upload"** endpoint
- Click "Try it out"
- Click "Choose Files" and select your PDF/DOCX files
- Click "Execute"

### 4. Query the Chatbot
- Click the **"POST /query"** endpoint
- Click "Try it out"
- In the request body, enter:
```json
{
  "query": "What is the CSE curriculum?",
  "top_k": 3
}
```
- Click "Execute"

### 5. View Results
The response will show:
- `response` - The most relevant document content
- `relevant_documents` - Top 3 matching documents with similarity scores
- `processing_time` - How long the query took

## ⚙️ API Endpoints

### GET /
Returns API overview and available endpoints

**Response:**
```json
{
  "message": "CSE Chatbot API",
  "endpoints": [...]
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /status
Get chatbot status

**Response:**
```json
{
  "loaded": true,
  "documents_count": 42,
  "knowledge_base_path": "knowledge_base.json",
  "model": "all-MiniLM-L6-v2"
}
```

### POST /upload
Upload documents (PDF, DOCX, TXT)

**Parameters:**
- `files` - File(s) to upload (multipart form data)

**Response:**
```json
{
  "message": "Files uploaded and processing started",
  "files": ["document.pdf"],
  "status": "processing"
}
```

### POST /query
Query the chatbot

**Request Body:**
```json
{
  "query": "Your question here",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "Your question here",
  "response": "Relevant content from documents...",
  "relevant_documents": [
    {
      "content": "Document text...",
      "metadata": {
        "source": "file.pdf",
        "type": "pdf",
        "chunk_size": 1000
      },
      "similarity": 0.85
    }
  ],
  "processing_time": 0.234
}
```

### DELETE /knowledge-base
Clear all knowledge base

**Response:**
```json
{
  "message": "Knowledge base cleared successfully"
}
```

### POST /reload-knowledge-base
Reload from saved file

**Response:**
```json
{
  "message": "Knowledge base reloaded",
  "documents_loaded": 42
}
```

## 🐛 Troubleshooting

### Issue: Port 8000 is already in use
**Solution:**
```bash
# Use a different port
python main.py --port 8001
```

### Issue: "Module not found" errors
**Solution:**
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Slow first request (model downloading)
**Solution:** The sentence-transformers model (~500 MB) downloads on first run. This is normal and happens only once.

### Issue: "No documents processed"
**Solution:**
1. Upload files first using POST /upload
2. Wait for processing to complete
3. Then query using POST /query

## 🔧 Advanced Configuration

### Change the Model
Edit `main.py` and modify:
```python
chatbot = CSEChatbot(model_name='all-MiniLM-L6-v2')  # Change this
```

Available models:
- `all-MiniLM-L6-v2` (lightweight, fast)
- `all-mpnet-base-v2` (better quality, slower)
- `paraphrase-MiniLM-L6-v2` (good balance)

### Adjust Similarity Threshold
Edit `chatbot_core.py`:
```python
if similarities.max() < 0.2:  # Change 0.2 to your threshold
    return "No information available"
```

### Change Chunk Size
Edit `chatbot_core.py`:
```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    # Adjust chunk_size and overlap as needed
```

## 📊 Performance Tips

1. **Use lightweight model** for faster responses
2. **Reduce chunk size** for faster processing
3. **Increase top_k** (default 3) for more thorough search
4. **Pre-process documents** to remove noise

## 📝 Example Use Cases

### University CSE Department Chatbot
```json
{
  "query": "What are the admission requirements for CSE?",
  "top_k": 5
}
```

### Course Information
```json
{
  "query": "Tell me about Data Structures course",
  "top_k": 3
}
```

### FAQ Bot
```json
{
  "query": "How do I register for courses?",
  "top_k": 3
}
```

## 🆘 Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify files are in the correct format (PDF, DOCX, TXT)
4. Check that documents are uploaded before querying
5. Try the `/docs` endpoint for interactive testing

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Server runs without errors (`python main.py`)
- [ ] Can access `http://localhost:8000/docs`
- [ ] Can upload documents
- [ ] Can query the chatbot
- [ ] Getting relevant responses

Once all checks pass, your chatbot is ready to use! 🎉
