# Automated Book Publication Workflow

A comprehensive system for automated book publication with AI-driven content processing, human-in-the-loop iterations, and reinforcement learning-based optimization.

## Features

### 1. Web Scraping & Screenshots
- **Playwright-based scraping**: Fetch content from web URLs with full browser automation
- **Screenshot capture**: Save visual representations of source content
- **RL-based reward system**: Optimize scraping strategies using reinforcement learning

### 2. AI Writing & Review Pipeline
- **AI Writer**: Transform scraped content using LLMs (Gemini/OpenAI)
- **AI Reviewer**: Quality assessment and improvement suggestions
- **Multi-agent coordination**: Seamless content flow between AI agents

### 3. Human-in-the-Loop Workflow
- **Iterative refinement**: Multiple rounds of human input for writers, reviewers, and editors
- **Version control**: Track all changes and iterations
- **Approval workflow**: Structured approval process before finalization

### 4. Advanced Features
- **Voice support**: Voice input/output for hands-free operation
- **Semantic search**: ChromaDB-powered content search and retrieval
- **RL-based inference**: Reinforcement learning for consistent data retrieval
- **Agentic API**: RESTful API for seamless integration

## Architecture

```
book_publication_workflow/
├── src/
│   ├── agents/           # AI agents (Writer, Reviewer, Editor)
│   ├── scraping/         # Web scraping and screenshot tools
│   ├── rl/              # Reinforcement learning components
│   ├── api/             # REST API endpoints
│   ├── database/        # ChromaDB integration
│   └── utils/           # Utility functions
├── config/              # Configuration files
├── data/               # Data storage
├── tests/              # Test suite
└── requirements.txt    # Python dependencies
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Run the system**:
   ```bash
   python src/main.py
   ```

## Usage Examples

### Basic Content Processing
```python
from src.workflow import BookPublicationWorkflow

workflow = BookPublicationWorkflow()
result = workflow.process_url("https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1")
```

### Human-in-the-Loop Iteration
```python
# Start iteration process
iteration = workflow.start_iteration(chapter_id="chapter_1")
# Human review and feedback
workflow.submit_human_feedback(iteration_id, feedback)
# Continue until approval
```

## API Endpoints

- `POST /api/scrape` - Scrape content from URL
- `POST /api/process` - Process content with AI agents
- `POST /api/iterate` - Start human-in-the-loop iteration
- `GET /api/search` - Semantic search content
- `POST /api/voice` - Voice input/output operations

## Configuration

Key configuration options in `config/settings.py`:
- LLM provider selection (Gemini/OpenAI)
- RL reward function parameters
- ChromaDB connection settings
- Voice processing options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details 