# 🤖 Automated Book Publication Workflow

A comprehensive AI-powered system for automated content creation, web scraping, and book publication with reinforcement learning optimization.

## 🚀 Features

### 🌐 **Web Scraping & Content Extraction**
- **Playwright-based scraping** with screenshot capture
- **Intelligent content extraction** with quality scoring
- **Metadata preservation** (title, author, date, etc.)
- **ChromaDB vector storage** for semantic search

### 🤖 **AI-Powered Content Processing**
- **AI Writer**: Content transformation and enhancement
- **AI Reviewer**: Quality assessment and feedback
- **AI Editor**: Final refinement and optimization
- **Multi-LLM Support**: OpenAI, Anthropic Claude, Google Gemini

### 🧠 **Reinforcement Learning**
- **RL Reward System**: Continuous learning from feedback
- **State Vector Optimization**: Dynamic content processing
- **Performance Tracking**: Quality metrics and improvements
- **Model Persistence**: Save and load trained models

### 🎤 **Voice Integration**
- **Text-to-Speech**: Content narration
- **Voice Commands**: Hands-free operation
- **Audio Processing**: Multiple format support

### 🔄 **Human-in-the-Loop**
- **Iteration System**: Feedback-driven improvements
- **Version Control**: Content versioning and tracking
- **Collaborative Workflow**: Team-based content creation

### 🌐 **API & CLI Interface**
- **FastAPI REST API**: Full programmatic access
- **CLI Mode**: Command-line processing
- **Web Interface**: Browser-based management
- **Real-time Processing**: Live status updates

## 📋 Prerequisites

- **Python 3.9+**
- **Git**
- **API Keys** for LLM providers (OpenAI, Anthropic, Google)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd book_publication_workflow-main
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
```bash
cp env.example .env
```

Edit `.env` with your API keys:
```env
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here

# Default LLM Provider (openai, anthropic, gemini)
DEFAULT_LLM_PROVIDER=anthropic

# Model Names
OPENAI_MODEL=gpt-3.5-turbo
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
GEMINI_MODEL=gemini-1.5-pro

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Voice Processing
VOICE_ENABLED=true
TTS_ENGINE=pyttsx3

# RL Configuration
RL_ENABLED=true
RL_LEARNING_RATE=0.001
RL_BATCH_SIZE=32
```

### 5. Initialize Data Directories
```bash
mkdir -p data/screenshots data/scraped_content data/processed_content data/voice_output data/iterations
mkdir -p models logs
```

## 🚀 Quick Start

### CLI Mode (Recommended for Testing)
```bash
# Process a single URL
python src/main.py --mode cli --url "https://example.com/article"

# Process with custom configuration
python src/main.py --mode cli --url "https://example.com/article" --style creative --provider anthropic
```

### API Server Mode
```bash
# Start the API server
python src/main.py --mode api --host 0.0.0.0 --port 8000

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

## 📖 Usage Examples

### 1. Basic Content Processing
```python
from src.workflow import BookPublicationWorkflow

# Initialize workflow
workflow = BookPublicationWorkflow()
workflow.initialize()

# Process a URL
result = await workflow.process_url("https://example.com/article")
print(f"Quality Score: {result.quality_score}")
print(f"Processing Time: {result.processing_time}")
```

### 2. Custom AI Configuration
```python
from src.agents.ai_agents import AgentOrchestrator

# Create orchestrator with custom config
orchestrator = AgentOrchestrator()
config = {
    "writing_style": "creative",
    "review_focus": ["clarity", "engagement"],
    "editing_focus": ["grammar", "style"],
    "custom_prompts": {
        "writer": "Transform this into a compelling blog post",
        "reviewer": "Focus on SEO optimization and readability"
    }
}

# Process content
processed = await orchestrator.process_content(scraped_content, config)
```

### 3. API Integration
```python
import requests

# Process URL via API
response = requests.post("http://localhost:8000/process", json={
    "url": "https://example.com/article",
    "config": {
        "writing_style": "professional",
        "provider": "anthropic"
    }
})

result = response.json()
print(f"Session ID: {result['session_id']}")
print(f"Quality Score: {result['quality_score']}")
```

### 4. Voice Processing
```python
from src.voice.voice_processor import VoiceProcessor

# Initialize voice processor
voice = VoiceProcessor()
voice.initialize_tts()

# Convert text to speech
audio_file = await voice.text_to_speech("Your processed content here")
print(f"Audio saved to: {audio_file}")
```

## 🔧 Configuration

### LLM Provider Settings
```python
# In config/settings.py
LLM_PROVIDERS = {
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "models": {
            "writer": "gpt-3.5-turbo",
            "reviewer": "gpt-3.5-turbo",
            "editor": "gpt-3.5-turbo"
        }
    },
    "anthropic": {
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "models": {
            "writer": "claude-3-5-sonnet-20241022",
            "reviewer": "claude-3-5-sonnet-20241022",
            "editor": "claude-3-5-sonnet-20241022"
        }
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "models": {
            "writer": "gemini-1.5-pro",
            "reviewer": "gemini-1.5-pro",
            "editor": "gemini-1.5-pro"
        }
    }
}
```

### RL Configuration
```python
# Reinforcement Learning settings
RL_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "memory_size": 10000,
    "gamma": 0.99,
    "epsilon": 0.1,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01
}
```

## 📊 API Endpoints

### Content Processing
- `POST /process` - Process a URL
- `GET /status/{session_id}` - Get processing status
- `GET /content/{content_id}` - Retrieve processed content

### Iteration Management
- `POST /iterate/{content_id}` - Start iteration
- `POST /feedback/{iteration_id}` - Submit feedback
- `GET /iterations/{content_id}` - Get iteration history

### Voice Processing
- `POST /voice/tts` - Convert text to speech
- `POST /voice/stt` - Convert speech to text

### Analytics
- `GET /stats` - Get system statistics
- `GET /performance` - Get RL performance metrics

## 🧪 Testing

### Run Basic Tests
```bash
python -m pytest tests/
```

### Test Specific Components
```bash
# Test web scraping
python tests/test_scraping.py

# Test AI agents
python tests/test_agents.py

# Test RL system
python tests/test_rl.py
```

## 📈 Performance Monitoring

### Quality Metrics
- **Content Quality Score**: 0-1 scale
- **Processing Time**: Seconds per URL
- **RL Reward**: Learning progress
- **API Success Rate**: Percentage of successful calls

### Logging
```bash
# View real-time logs
tail -f logs/workflow.log

# Check specific component logs
tail -f logs/ai_agents.log
tail -f logs/web_scraper.log
```

## 🔒 Security Considerations

### API Key Management
- Store API keys in `.env` file (not in version control)
- Use environment variables for production
- Rotate keys regularly
- Monitor API usage and costs

### Data Privacy
- Scraped content is stored locally
- No data is sent to external services (except LLM APIs)
- ChromaDB data is encrypted at rest
- Implement proper access controls

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/main.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Setup
```bash
# Install production dependencies
pip install -r requirements.txt

# Set production environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Start with process manager
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.main:app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/
isort src/

# Run tests with coverage
pytest --cov=src tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**1. API Key Errors**
```bash
# Check your .env file
cat .env

# Verify API keys are valid
python -c "import os; print(os.getenv('ANTHROPIC_API_KEY')[:10] + '...')"
```

**2. Model Loading Errors**
```bash
# Clear old model files
rm -rf models/*.pth

# Restart the system
python src/main.py --mode cli --url "https://example.com/article"
```

**3. ChromaDB Issues**
```bash
# Reset database
rm -rf chroma_db/
python -c "from src.database.chroma_manager import ChromaManager; ChromaManager().initialize_chroma()"
```

### Getting Help
- 📧 Email: support@bookworkflow.com
- 💬 Discord: [Join our community](https://discord.gg/bookworkflow)
- 📖 Documentation: [Full docs](https://docs.bookworkflow.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Web scraping with screenshots
- ✅ AI content processing
- ✅ RL optimization
- ✅ Voice integration
- ✅ API/CLI interfaces

### Phase 2 (Next)
- 🔄 Multi-language support
- 🔄 Advanced content templates
- 🔄 Social media integration
- 🔄 Automated publishing
- 🔄 Advanced analytics

### Phase 3 (Future)
- 📅 Real-time collaboration
- 📅 Advanced AI models
- 📅 Blockchain integration
- 📅 Mobile app
- 📅 Enterprise features

---

**Made with ❤️ by the Book Publication Workflow Team**

*Transform your content creation workflow with AI-powered automation!* 