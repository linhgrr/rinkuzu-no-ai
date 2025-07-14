# 🤖 AI Service - Educational Assistant Platform

A comprehensive FastAPI-based microservice for educational applications, featuring **Rin-chan**, an intelligent RAG-based tutor system with advanced document processing and contextual question answering.

## 🌟 Key Features

- **📚 RAG Tutoring**: Upload learning materials and ask intelligent questions
- **🔍 Smart Search**: Vector search with reranking for accurate answers  
- **💬 Multi-turn Chat**: Contextual conversations with AI tutor
- **📄 Document Processing**: Support for PDF, DOCX, and other formats
- **🌐 Multi-language**: Vietnamese and English support
- **⚡ High Performance**: Optimized embedding and generation pipeline
- **🏗️ Clean Architecture**: OOP design with dependency injection
- **🔌 Plug-and-play**: Easy to swap AI models and vector stores

## 🛠️ Technology Stack

- **🧠 AI Models**: Google Gemini 2.0 Flash (with automatic fallback)
- **🔤 Embeddings**: GTE Multilingual Base & Reranker (Hit@1 > 84%)
- **🗄️ Vector DB**: ChromaDB with persistent storage
- **🚀 Framework**: FastAPI with async/await support
- **📝 File Processing**: 
  - **PDF**: PyMuPDF (advanced layout preservation)
  - **DOCX**: python-docx (tables, formatting)
  - **PPTX**: python-pptx (slides, notes)
  - **Images**: Gemini Vision (OCR, analysis)
- **🔒 Security**: Input validation, API key auth, CORS
- **📊 Monitoring**: Comprehensive logging with loguru

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd ai_service

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file based on the example:

```bash
# Copy the example file
cp env.example .env

# Then edit .env with your settings
```

**Required Configuration:**
```bash
# Google Gemini API Keys (get from https://ai.google.dev/)
GEMINI_KEYS=your_gemini_key_1,your_gemini_key_2,your_gemini_key_3

# File upload settings
ALLOWED_FILE_TYPES=pdf,docx,pptx,png,jpg,jpeg,gif,bmp,webp
MAX_FILE_SIZE=52428800  # 50MB

# Vector database
VECTOR_DB_PATH=./vector_db
```

**Optional Settings:**
```bash
DEBUG=true
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
MAX_RETRIEVAL_DOCS=10
RERANKER_TOP_K=3
```

### 3. Run the Service

**Option 1: Quick Start (Recommended)**
```bash
# Use the startup script with automatic checks
python start.py
```

**Option 2: Manual Start**
```bash
# Development mode with auto-reload
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 API Usage Examples

### Upload Learning Material

**Upload PDF/DOCX/PPTX:**
```bash
curl -X POST "http://localhost:8000/v1/tutor/upload-material" \
  -F "file=@lecture_notes.pdf" \
  -F "subject_id=computer_science_101" \
  -F "metadata={\"course\": \"CS101\", \"chapter\": \"1\"}"
```

**Upload Images with Gemini Vision:**
```bash
curl -X POST "http://localhost:8000/v1/tutor/upload-material" \
  -F "file=@diagram.png" \
  -F "subject_id=computer_science_101" \
  -F "metadata={\"type\": \"diagram\", \"topic\": \"CPU architecture\"}"
```

**Upload PowerPoint Presentation:**
```bash
curl -X POST "http://localhost:8000/v1/tutor/upload-material" \
  -F "file=@slides.pptx" \
  -F "subject_id=computer_science_101" \
  -F "metadata={\"lecture\": \"Introduction to AI\"}"
```

### Ask Rin-chan a Question

```bash
curl -X POST "http://localhost:8000/v1/tutor/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "CPU hoạt động như thế nào?",
    "subject_id": "computer_science_101",
    "use_reranking": true
  }'
```

### Multi-turn Conversation

```bash
curl -X POST "http://localhost:8000/v1/tutor/ask-question" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Cho ví dụ về cache miss?",
    "subject_id": "computer_science_101",
    "chat_history": [
      {"role": "user", "content": "CPU hoạt động như thế nào?"},
      {"role": "assistant", "content": "CPU xử lý dữ liệu thông qua các chu kỳ fetch, decode, execute..."}
    ]
  }'
```

### List Available Subjects

```bash
curl -X GET "http://localhost:8000/v1/tutor/subjects"
```

### General AI Chat

```bash
curl -X POST "http://localhost:8000/v1/ai/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "temperature": 0.7
  }'
```

## 🏗️ Architecture Overview

### Core Components

```
ai_service/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── app/
│   ├── core/
│   │   ├── config.py      # Environment configuration
│   │   └── interfaces/    # Abstract interfaces
│   │       ├── ai_service.py
│   │       ├── file_loader.py
│   │       └── vector_store.py
│   ├── services/
│   │   ├── ai/
│   │   │   ├── gemini_service.py      # Gemini AI implementation
│   │   │   └── rag_tutor_service.py   # Main RAG service
│   │   ├── vector/
│   │   │   ├── embedding_service.py   # GTE embeddings
│   │   │   └── chroma_store.py        # ChromaDB integration
│   │   └── file_loaders/
│   │       ├── pdf_loader.py          # PDF processing
│   │       ├── docx_loader.py         # DOCX processing
│   │       └── factory.py             # Loader factory
│   ├── api/v1/
│   │   ├── routes/
│   │   │   ├── tutor.py   # RAG tutoring endpoints
│   │   │   └── ai.py      # General AI endpoints
│   │   └── __init__.py    # API router
│   ├── models/
│   │   ├── requests.py    # Pydantic request models
│   │   └── responses.py   # Pydantic response models
│   └── utils/
│       ├── dependencies.py    # FastAPI dependencies
│       └── key_rotation.py    # API key management
```

### Design Patterns

1. **Interface Segregation**: Abstract interfaces for all major components
2. **Dependency Injection**: Services injected via FastAPI Depends
3. **Factory Pattern**: Automatic file loader selection
4. **Strategy Pattern**: Pluggable AI services and vector stores
5. **Observer Pattern**: Key rotation with health monitoring

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_KEYS` | ✅ | - | Comma-separated Gemini API keys |
| `DEBUG` | ❌ | false | Enable debug mode |
| `HOST` | ❌ | 0.0.0.0 | Server host |
| `PORT` | ❌ | 8000 | Server port |
| `LOG_LEVEL` | ❌ | INFO | Logging level |
| `VECTOR_DB_PATH` | ❌ | ./vector_db | ChromaDB storage path |
| `EMBEDDING_MODEL` | ❌ | Alibaba-NLP/gte-multilingual-base | HuggingFace model |
| `RERANKER_MODEL` | ❌ | Alibaba-NLP/gte-multilingual-reranker-base | Reranker model |
| `CHUNK_SIZE` | ❌ | 1000 | Text chunk size |
| `CHUNK_OVERLAP` | ❌ | 100 | Chunk overlap |
| `MAX_RETRIEVAL_DOCS` | ❌ | 10 | Max documents to retrieve |
| `RERANKER_TOP_K` | ❌ | 3 | Top documents after reranking |
| `MAX_FILE_SIZE` | ❌ | 52428800 | Max upload size (50MB) |
| `ALLOWED_FILE_TYPES` | ❌ | pdf,docx,txt | Supported file types |
| `API_KEY` | ❌ | - | Optional API authentication |
| `CORS_ORIGINS` | ❌ | * | CORS allowed origins |

### RAG Pipeline Configuration

The system uses a sophisticated RAG pipeline:

1. **Document Processing**: Files are chunked with configurable size and overlap
2. **Embedding**: GTE Multilingual Base creates semantic vectors
3. **Storage**: ChromaDB stores documents with metadata filtering
4. **Retrieval**: Vector similarity search finds relevant chunks
5. **Reranking**: GTE Reranker improves result quality (Hit@1 > 84%)
6. **Generation**: Gemini 2.0 Flash generates contextual answers

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "main.py"]
```

### Environment Variables for Production

```bash
DEBUG=false
LOG_LEVEL=INFO
API_KEY=your_production_api_key
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
VECTOR_DB_PATH=/app/data/vector_db
```

## 🔌 Extending the System

### Adding New AI Services

1. Implement the `AIService` interface:

```python
from app.core.interfaces.ai_service import AIService, AIResponse

class CustomAIService(AIService):
    async def generate_text(self, prompt: str, **kwargs) -> AIResponse:
        # Your implementation
        pass
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> AIResponse:
        # Your implementation
        pass
```

2. Register in dependencies:

```python
# app/utils/dependencies.py
@lru_cache()
def get_custom_ai_service() -> CustomAIService:
    return CustomAIService()
```

### Adding New File Loaders

```python
from app.core.interfaces.file_loader import FileLoader, DocumentChunk

class CustomLoader(FileLoader):
    async def load(self, file_stream: BinaryIO, filename: str) -> List[DocumentChunk]:
        # Your implementation
        pass
    
    def supports_file_type(self, file_extension: str) -> bool:
        return file_extension in ['custom']

# Register in factory
from app.services.file_loaders.factory import file_loader_factory
file_loader_factory.register_loader('custom', CustomLoader)
```

## 📊 Monitoring and Logging

### Health Checks

- `GET /health` - Overall service health
- `GET /v1/tutor/health` - RAG-specific health
- `GET /v1/ai/statistics` - AI service statistics

### Logging

The service uses structured logging with loguru:

```python
# Configure custom logging
from loguru import logger

logger.add("ai_service.log", rotation="1 day", retention="30 days")
```

### Performance Metrics

- Request/response times
- API key usage and rotation
- Embedding cache hit rates
- Vector search performance
- Model success/failure rates

## 🤝 Integration Examples

### Next.js Frontend Integration

```typescript
// Frontend integration example
const uploadDocument = async (file: File, subjectId: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('subject_id', subjectId);
  
  const response = await fetch('/api/ai/v1/tutor/upload-material', {
    method: 'POST',
    body: formData,
  });
  
  return await response.json();
};

const askQuestion = async (question: string, subjectId: string) => {
  const response = await fetch('/api/ai/v1/tutor/ask-question', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, subject_id: subjectId }),
  });
  
  return await response.json();
};
```

### Proxy Configuration

```javascript
// next.config.js
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/ai/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
};
```

## 🔍 Troubleshooting

### Common Issues

1. **Service won't start**: Check GEMINI_KEYS configuration
2. **Upload fails**: Verify file size and type restrictions
3. **No context found**: Ensure documents are uploaded for the subject
4. **Slow responses**: Check embedding model loading and cache

### Debug Mode

Enable debug mode for detailed logging:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

## 📝 License

This project is part of the CK Quiz educational platform.

## 🤝 Contributing

1. Follow the established architecture patterns
2. Add comprehensive tests for new features
3. Update documentation and type hints
4. Ensure all services implement proper interfaces

---

**Built with ❤️ for education. Rin-chan is ready to help students learn! 🎓** 

## 🚀 Performance Tuning for Concurrent Uploads

### Environment Variables for Optimization

```bash
# Concurrent Processing Control
MAX_CONCURRENT_EMBEDDINGS=3        # Max parallel embedding operations
EMBEDDING_TIMEOUT_SECONDS=300      # Timeout for embedding operations (5 min)
FAISS_THREAD_POOL_WORKERS=0        # Thread pool size (0 = auto)
BATCH_SIZE_DOCUMENTS=10             # Process documents in batches

# Memory Management
CHUNK_SIZE=1000                     # Text chunk size for processing
CHUNK_OVERLAP=200                   # Overlap between chunks
MAX_CHUNKS_PER_DOCUMENT=50          # Limit chunks per document
```

### Tuning Guidelines

#### For High Volume Uploads (10+ files simultaneously):
```bash
MAX_CONCURRENT_EMBEDDINGS=2        # Reduce to prevent resource contention
EMBEDDING_TIMEOUT_SECONDS=600      # Increase timeout for large files
FAISS_THREAD_POOL_WORKERS=6        # More workers for CPU-intensive tasks
BATCH_SIZE_DOCUMENTS=5              # Smaller batches to reduce memory usage
```

#### For Large Files (>10MB each):
```bash
EMBEDDING_TIMEOUT_SECONDS=600      # Longer timeout for large files
CHUNK_SIZE=800                      # Smaller chunks for faster processing
MAX_CHUNKS_PER_DOCUMENT=100         # Allow more chunks for large docs
```

#### For Limited Resources (Low RAM/CPU):
```bash
MAX_CONCURRENT_EMBEDDINGS=1        # Process one at a time
FAISS_THREAD_POOL_WORKERS=2        # Minimal workers
BATCH_SIZE_DOCUMENTS=3              # Very small batches
CHUNK_SIZE=500                      # Smaller chunks
```

### Monitoring Performance

Monitor the logs for these indicators:

- ✅ `Processing batch X/Y` - Batch processing progress
- ⏱️ `add_documents timed out after X seconds` - Increase timeout
- 🔄 `Selected X chunks (rerank=on/off)` - Processing efficiency
- 🚫 `Semaphore blocking` - Too many concurrent operations

### Optimization Tips

1. **Start Conservative**: Begin with default values and adjust based on actual performance
2. **Monitor Resources**: Watch CPU, memory, and disk I/O during uploads
3. **Test Incrementally**: Change one parameter at a time to measure impact
4. **Consider Hardware**: Adjust based on your server's CPU cores and RAM 