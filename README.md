# RAG-PDF-Chat-App

A Streamlit-based application that allows users to chat with PDF documents using multiple AI providers through RAG (Retrieval-Augmented Generation).

## Features

- Support for multiple AI providers:
  - OpenAI
  - Google Gemini
  - Anthropic Claude
  - Grok
- PDF text extraction with multiple fallback methods
- Intelligent text chunking with overlap
- Semantic search using FAISS
- Configurable chunk size and overlap
- Chat history management
- Export chat history functionality
- Customizable UI settings

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ArekatlaNishanthchowdary/RAG-PDF-Chat-App.git
cd RAG-PDF-Chat-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables (optional):
- `EMBEDDING_MODEL`: Name of the embedding model to use
- `CHUNK_SIZE`: Size of text chunks
- `CHUNK_OVERLAP`: Overlap between chunks
- `DEFAULT_PROVIDER`: Default AI provider
- Other provider-specific settings

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select your preferred AI provider in the sidebar
2. Enter your API key for the selected provider
3. Upload a PDF document
4. Start chatting with the document content

## Configuration

The application can be configured through:
- Environment variables
- `config.py` file
- Streamlit sidebar settings

## License

MIT License with Gemini API

A comprehensive **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and chat with their content using Google's Gemini AI. Built with Streamlit for a user-friendly web interface.

## 🌟 Features

### Core Functionality
- **PDF Processing**: Upload and extract text from multi-page PDFs with fallback mechanisms
- **Intelligent Text Chunking**: Semantic splitting with configurable chunk size and overlap
- **Vector Embeddings**: Uses SentenceTransformers (all-MiniLM-L6-v2) for high-quality embeddings
- **Efficient Search**: FAISS vector database for fast similarity search
- **AI-Powered Chat**: Google Gemini API integration for natural language responses
- **Source Attribution**: Shows which document sections were used for each response

### User Interface
- **Modern Web Interface**: Clean, responsive Streamlit design
- **Real-time Chat**: Interactive chat interface with message history
- **Configurable Settings**: Adjustable chunk size, overlap, and retrieval parameters
- **Secure API Key Handling**: Safe storage in session state
- **Export Functionality**: Download chat history as JSON

### Advanced Features
- **Multiple PDF Support**: Switch between different uploaded documents
- **Conversation Memory**: Maintains context across chat exchanges
- **Error Handling**: Graceful handling of API limits and failures
- **Progress Tracking**: Visual feedback during PDF processing
- **Source Highlighting**: View relevant document chunks for each response

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Setup
1. **Configure API Key**: Enter your Gemini API key in the sidebar
2. **Upload PDF**: Choose a PDF document to analyze
3. **Start Chatting**: Ask questions about your document content

## 📋 Dependencies

```
streamlit==1.29.0          # Web framework
google-generativeai==0.3.2 # Gemini API client
PyPDF2==3.0.1             # PDF text extraction (fallback)
pdfplumber==0.10.3        # PDF text extraction (primary)
sentence-transformers==2.2.2 # Text embeddings
faiss-cpu==1.7.4          # Vector similarity search
numpy==1.24.3             # Numerical computing
pandas==2.0.3             # Data manipulation
python-dotenv==1.0.0      # Environment variables
```

## 🏗️ Architecture

### Core Components

1. **PDFProcessor**: Handles PDF text extraction with multiple methods for reliability
2. **TextChunker**: Intelligent text splitting with sentence boundary detection
3. **EmbeddingManager**: Manages SentenceTransformer model and FAISS index
4. **GeminiChat**: Handles Gemini API interactions and prompt engineering
5. **Configuration**: Flexible settings management with environment variable support

### Data Flow

```
PDF Upload → Text Extraction → Text Chunking → Embedding Generation → FAISS Index
                                                                            ↓
User Query → Query Embedding → Similarity Search → Context Retrieval → Gemini API → Response
```

## ⚙️ Configuration

### Default Settings
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Embedding Model**: all-MiniLM-L6-v2
- **Retrieval Count**: 5 chunks per query
- **Gemini Model**: gemini-pro

### Environment Variables
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CHUNKS_RETRIEVE=5
GEMINI_MODEL=gemini-pro
MAX_CHAT_HISTORY=10
```

## 💡 Usage Examples

### Basic Usage
1. Upload a research paper or technical document
2. Ask: "What is the main contribution of this paper?"
3. Follow up: "Can you explain the methodology in detail?"

### Advanced Queries
- "Compare the results in sections 3 and 4"
- "What are the limitations mentioned in the conclusion?"
- "Summarize the key findings from the experiment"

## 🔧 Customization

### Adjusting Performance
- **Increase chunk size** for longer context but slower processing
- **Reduce overlap** for faster processing but potential information loss
- **Adjust retrieval count** based on document complexity

### API Configuration
- Monitor Gemini API usage and quotas
- Implement rate limiting for production use
- Consider using different Gemini models based on requirements

## 🧪 Testing

Run the test suite:
```bash
python -m pytest test_app.py -v
```

Or run individual test classes:
```bash
python test_app.py
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
1. **Docker**: Create a Dockerfile for containerization
2. **Cloud Platforms**: Deploy to Streamlit Cloud, Heroku, or AWS
3. **Environment**: Set up environment variables for production
4. **Monitoring**: Implement logging and error tracking

## 📊 Performance Optimization

### For Large Documents
- Enable text preprocessing and cleaning
- Implement document caching mechanisms
- Use more efficient embedding models if needed
- Consider document summarization for very long texts

### For High Traffic
- Implement connection pooling for Gemini API
- Add request queuing and rate limiting
- Cache frequent queries and responses
- Use CDN for static assets

## 🔒 Security Considerations

- **API Key Security**: Never commit API keys to version control
- **Input Validation**: Sanitize user inputs and file uploads
- **Rate Limiting**: Implement API call limits to prevent abuse
- **Session Management**: Secure session state handling

## 🐛 Troubleshooting

### Common Issues

**PDF Processing Fails**
- Try different PDF extraction method
- Ensure PDF is not password-protected or corrupted
- Check file size limits

**Embedding Model Loading**
- Verify internet connection for model download
- Check available disk space
- Clear model cache if needed

**Gemini API Errors**
- Verify API key validity
- Check API quota and billing
- Review request rate limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

## 📄 License

This project is open source. Please ensure you comply with the terms of service for:
- Google Gemini API
- SentenceTransformers library
- All other dependencies

## 🔄 Updates and Roadmap

### Planned Features
- [ ] Multiple document comparison
- [ ] Advanced search filters
- [ ] Response confidence scoring
- [ ] Document annotation system
- [ ] Multi-language support
- [ ] Integration with cloud storage

### Version History
- **v1.0.0**: Initial release with core RAG functionality
- **v1.1.0**: Enhanced UI and error handling
- **v1.2.0**: Advanced configuration options

---

**Built with ❤️ using Streamlit, Gemini AI, and modern RAG techniques**#
