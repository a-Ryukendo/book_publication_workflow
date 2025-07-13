# Book Publication Workflow

A system for automated content creation, web scraping, and book publication using AI and reinforcement learning.

## Features

- Web scraping and content extraction (Playwright, screenshots)
- Content scoring and metadata
- ChromaDB vector storage for semantic search
- AI-based writing, reviewing, editing (OpenAI, Anthropic, Google Gemini)
- Reinforcement learning for optimization
- Voice support (text-to-speech, audio input)
- Human feedback and version control
- FastAPI REST API, CLI, and web interface

## Prerequisites

- Python 3.9+
- Git
- API keys for OpenAI, Anthropic, Google

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd book_publication_workflow-main
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Setup environment variables:
    ```bash
    cp env.example .env
    ```
    Edit `.env` with your API keys and configuration.

5. Initialize directories:
    ```bash
    mkdir -p data/screenshots data/scraped_content data/processed_content data/voice_output data/iterations
    mkdir -p models logs
    ```

## Quick Start

- To process a single URL in CLI mode:
    ```bash
    python src/main.py --mode cli --url <target-url> --session <session-name>
    ```
- To run the API server:
    ```bash
    python src/main.py --mode api
    ```
    See API docs at `/docs` when the server is running.

## Configuration

Edit `.env` for model keys, database paths, voice options, RL settings, and workflow config.

## Contributing

Fork the repo, create a branch, add tests, and open a pull request.

## License

MIT
