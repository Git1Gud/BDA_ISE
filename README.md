# 🎓 QA-RAG Study Material Generation System

A comprehensive implementation of the QA-RAG (Question-Answering Retrieval-Augmented Generation) approach based on the MDPI paper, featuring advanced study material generation with hybrid search capabilities.

## 🌟 Features

### Core Capabilities
- **Hybrid RAG Search**: Combines dense vector similarity with sparse BM25 retrieval
- **Study Material Generation**: Automatically creates comprehensive course materials
- **Marp Slide Generation**: Converts content to presentation-ready slides
- **PDF Processing**: Extracts and processes syllabus and reference materials

### QA-RAG Evaluation (MDPI Paper Implementation)
1. **Noise Robustness**: Tests system performance with varying retrieval sizes (k)
2. **Knowledge Gap Detection**: Identifies when insufficient information is available
3. **External Truth Integration**: Uses retrieved knowledge over parametric memory

### Interactive Dashboard
- **Streamlit Web Interface**: User-friendly interface for all features
- **Real-time Testing**: Interactive QA testing with confidence scoring
- **Performance Analytics**: Comprehensive evaluation metrics and visualizations
- **File Upload Support**: Easy PDF upload and processing

### Logging & Monitoring
- **Colored Console Logging**: Enhanced logging with color-coded messages
  - 🟢 INFO: General information (green)
  - 🟡 WARNING: Warnings and non-critical issues (yellow)
  - 🔴 ERROR: Errors and exceptions (red)
  - 🔵 DEBUG: Detailed debugging information (cyan)
- **Structured Logging**: Consistent log format with timestamps and context
- **Configurable Log Levels**: Adjust verbosity based on needs

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Qdrant vector database (local or cloud)
- Google Gemini API key
- Optional: Groq API key, MongoDB

### Installation

1. **Clone and Setup**:
```bash
cd "d:\College\College\Sem7\BDA\ISE"
```

2. **Install Dependencies**:
```bash
pip install -e .
```

3. **Configure Environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Start Qdrant** (if running locally):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

5. **Run the Application**:
```bash
python run_app.py
```

6. **Open Browser**: Navigate to `http://localhost:8501`

## 📖 Usage

### Colored Logging

The system includes a comprehensive colored logging system for better debugging and monitoring:

```python
from logger import logger

# Different log levels with colors
logger.info("Application started successfully")      # Green
logger.warning("Configuration issue detected")      # Yellow
logger.error("Failed to process request")           # Red
logger.debug("Processing step completed")           # Cyan
```

### Running the Application

```bash
# Run the main application
python app.py

# Test the logging system
python test_logging.py
``` Guide

### 1. System Initialization
- Start the application and initialize the RAG system using the sidebar
- Choose between hybrid search (recommended) or pure vector search

### 2. Document Upload
- Upload your course syllabus (PDF format)
- Upload reference materials (textbooks, papers, etc.)
- The system will automatically process and index the content

### 3. Study Material Generation
- Enter a topic or subject area
- Configure generation options (diagrams, slide count)
- Generate comprehensive study materials with automatic slide formatting

### 4. QA-RAG Testing
- Test the three critical QA-RAG abilities:
  - **Noise Robustness**: Vary retrieval size to test noise handling
  - **Knowledge Gap Detection**: Test recognition of insufficient information
  - **External Truth Integration**: Verify use of external vs parametric knowledge

### 5. Performance Evaluation
- View comprehensive metrics and analytics
- Export results for further analysis
- Track system performance across different test scenarios

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  StudyMaterialRAG │────│  Qdrant Vector  │
│                 │    │                  │    │    Database     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌──────────────────┐
                       │   LLM Models     │
                       │ • Google Gemini  │
                       │ • Groq (optional)│
                       └──────────────────┘
```

### Key Classes
- **StudyMaterialRAG**: Main RAG system with hybrid search
- **AnswerResponse**: Structured response model with confidence scoring
- **Topic**: Syllabus topic extraction and organization
- **FastEmbedSparse**: Custom BM25/TF-IDF sparse embeddings

### Hybrid Search Implementation
- **Dense Vectors**: Semantic similarity using sentence transformers
- **Sparse Vectors**: Keyword matching using TF-IDF/BM25
- **Fusion**: Reciprocal Rank Fusion (RRF) combining both approaches

## 🧪 QA-RAG Methodology

Based on the MDPI research paper, this implementation evaluates three critical abilities:

### 1. Noise Robustness
- Tests system performance with different retrieval sizes (k=1,2,3,5,8)
- Measures confidence degradation with increased noise
- Implements automated testing across multiple question sets

### 2. Knowledge Gap Detection
- Uses confidence thresholds to detect insufficient information
- Implements proper refusal mechanisms for unanswerable questions
- Provides detailed reasoning for gap detection decisions

### 3. External Truth Integration
- Prioritizes retrieved information over model's parametric knowledge
- Tests with questions requiring external context
- Validates proper source attribution and reasoning

## 📊 Evaluation Metrics

- **Accuracy**: Percentage of correct answers
- **Confidence Scores**: System certainty in responses
- **Noise Robustness**: Performance degradation analysis
- **Gap Detection Rate**: Success in identifying unanswerable questions
- **Response Quality**: Answer length and coherence metrics

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_API_KEY=your_google_api_key

# Optional
GROQ_API_KEY=your_groq_key
QDRANT_URL=http://localhost:6333
MONGO_URI=mongodb://localhost:27017/
```

### System Parameters
- **Chunk Size**: 1000 tokens (configurable)
- **Chunk Overlap**: 200 tokens
- **Default K**: 5 retrieved documents
- **Confidence Threshold**: 0.7
- **RRF Parameter**: 60 (for hybrid fusion)

## 📁 Project Structure

```
ISE/
├── main.py                 # Streamlit application
├── run_app.py             # Application launcher
├── temp_folder/
│   ├── temp.py           # RAG system implementation
│   └── aptemp.py         # Additional utilities
├── pyproject.toml         # Dependencies
├── .env.example          # Configuration template
└── README.md             # This file
```

## 🔬 Research Implementation

This project implements the QA-RAG methodology from:
> "QA-RAG: Question Answering via Retrieval Augmented Generation" - MDPI Journal

Key innovations implemented:
- Hybrid retrieval with sparse-dense fusion
- Confidence-based knowledge gap detection
- Multi-dimensional evaluation framework
- Interactive testing and validation tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MDPI research paper authors for QA-RAG methodology
- LangChain community for RAG framework
- Qdrant team for vector database technology
- Streamlit team for the web framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section below
2. Review the configuration guide
3. Open an issue on the repository

## 🔧 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure all dependencies are installed
pip install -e .
```

**Qdrant Connection**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/health
```

**API Key Issues**:
```bash
# Verify your .env file has the correct keys
cat .env
```

**Memory Issues**:
- Reduce chunk size in configuration
- Use smaller embedding models
- Limit the number of test questions

### Performance Optimization

1. **Use Local Qdrant**: Faster than cloud for development
2. **Enable Hybrid Search**: Better retrieval quality
3. **Tune Confidence Thresholds**: Based on your use case
4. **Optimize Chunk Size**: Balance between context and performance

---

**Built with ❤️ for advancing educational technology through AI**