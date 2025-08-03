# AI Pipeline with LangGraph, LangChain, and LangSmith

A simple AI pipeline demonstrating weather API integration and PDF RAG (Retrieval-Augmented Generation) using LangGraph for orchestration, LangChain for LLM operations, and LangSmith for evaluation.

## 🚀 Features

- **Intent Classification**: Automatically determines whether queries are about weather or PDF content
- **Weather API Integration**: Real-time weather data from OpenWeatherMap
- **PDF RAG System**: Upload and query PDF documents using vector embeddings
- **Vector Database**: Qdrant for storing and retrieving embeddings
- **LangGraph Orchestration**: Conditional workflow based on query intent
- **LangSmith Evaluation**: Built-in tracing and evaluation
- **Streamlit UI**: Interactive chat interface
- **Comprehensive Testing**: Unit tests for all components

## 🏗️ Architecture

```
User Query → Intent Classification → Weather API OR PDF Retrieval → LLM Response
```

### LangGraph Workflow

1. **classify_intent**: Determines if query is about weather or PDF content
2. **fetch_weather**: Calls OpenWeatherMap API for weather queries
3. **retrieve_from_pdf**: Searches vector database for PDF queries
4. **generate_response**: Uses LLM to create final response

## 📦 Installation

### Prerequisites

- Python 3.8+
- Qdrant server running locally or accessible remotely
- API keys for OpenAI, OpenWeatherMap, and LangSmith

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ai_pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Start Qdrant server**
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# Follow instructions at https://qdrant.tech/documentation/quick-start/
```

5. **Run the application**
```bash
streamlit run streamlit_app.py
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings | Yes |
| `OPENWEATHERMAP_API_KEY` | OpenWeatherMap API key | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key for evaluation | Optional |
| `LANGSMITH_TRACING` | Enable LangSmith tracing (true/false) | Optional |
| `LANGSMITH_PROJECT` | LangSmith project name | Optional |
| `QDRANT_URL` | Qdrant server URL | Optional (default: localhost:6333) |

### Getting API Keys

1. **OpenAI**: Visit [OpenAI API](https://platform.openai.com/api-keys)
2. **OpenWeatherMap**: Visit [OpenWeatherMap API](https://openweathermap.org/api)
3. **LangSmith**: Visit [LangSmith](https://smith.langchain.com/)

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_weather.py -v
```

### Test Coverage

- **Weather Service**: API calls, error handling, response formatting
- **PDF Service**: Text extraction, chunking, embedding creation
- **Graph Workflow**: Intent classification, conditional routing
- **Integration Tests**: End-to-end pipeline testing

## 🖥️ Usage

### Streamlit UI

1. **Upload PDF**: Use the sidebar to upload and process PDF documents
2. **Ask Questions**: Type queries in the chat interface
3. **View Details**: Expand response details to see processing metadata

### Sample Queries

**Weather Queries:**
- "What's the weather in London?"
- "How's the temperature in New York?"
- "Is it raining in Tokyo?"

**PDF Queries:**
- "What is the main topic of the document?"
- "Summarize the key points"
- "What are the conclusions?"

### Programmatic Usage

```python
from src.graph.graph import AIProcessingGraph

# Initialize the graph
graph = AIProcessingGraph()

# Process a query
result = graph.process_query("What's the weather in Paris?")
print(result["response"])
```

## 📊 LangSmith Evaluation

The pipeline includes built-in LangSmith tracing for:

- **Query Processing Time**: Track latency for different query types
- **Intent Classification Accuracy**: Monitor classification performance
- **Response Quality**: Evaluate LLM output quality
- **Error Tracking**: Monitor and debug failures

### Viewing Results

1. Log into [LangSmith](https://smith.langchain.com/)
2. Select your project (configured in `LANGSMITH_PROJECT`)
3. View traces and evaluation metrics

### Sample LangSmith Metrics

- **Latency**: Average response time by query type
- **Token Usage**: Track OpenAI API consumption
- **Error Rate**: Monitor system reliability
- **User Satisfaction**: Custom evaluation metrics

## 🔍 Code Structure

```
ai_pipeline/
├── src/
│   ├── config.py              # Configuration management
│   ├── graph/
│   │   ├── nodes.py           # LangGraph node functions
│   │   └── graph.py           # Main graph implementation
│   ├── services/
│   │   ├── weather_service.py # Weather API integration
│   │   ├── pdf_service.py     # PDF processing
│   │   └── vector_store.py    # Qdrant operations
│   └── utils/
│       └── helpers.py         # Utility functions
├── tests/                     # Comprehensive test suite
├── streamlit_app.py          # UI application
└── requirements.txt          # Dependencies
```

## 🚨 Error Handling

The pipeline includes robust error handling for:

- **API Failures**: Weather API timeouts, invalid responses
- **PDF Processing**: Corrupted files, unsupported formats
- **Vector Database**: Connection issues, storage failures
- **LLM Errors**: Rate limits, model unavailable

## 🔒 Security Notes

- API keys are loaded from environment variables
- No sensitive data is logged
- Vector database can be configured with authentication
- Rate limiting on API calls

## 🚀 Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Production Deployment
- Configure environment variables
- Set up Qdrant cluster
- Use production-grade ASGI server
- Implement proper logging and monitoring

## 🛠️ Customization

### Adding New Intents
1. Update `classify_intent` node in `nodes.py`
2. Add new processing node
3. Update graph routing logic
4. Add corresponding tests

### Custom Embeddings
- Modify `EMBEDDING_MODEL` in config
- Update vector dimensions in Qdrant configuration

### Different LLM Models
- Change `MODEL_NAME` in config
- Adjust temperature and other parameters

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Vector Database Connection**
   - Ensure Qdrant is running on the correct port
   - Check firewall settings

2. **API Key Issues**
   - Verify all required keys are set
   - Check key permissions and quotas

3. **PDF Processing Errors**
   - Ensure PDF files are not password-protected
   - Check file size limits

4. **LangSmith Tracing**
   - Verify API key and project configuration
   - Check network connectivity

### Getting Help

- Check the [Issues](../../issues) section
- Review LangChain/LangGraph documentation
- Contact the development team

---

**Built with ❤️ using LangChain, LangGraph, and LangSmith**