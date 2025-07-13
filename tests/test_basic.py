import pytest
import asyncio
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models import ScrapedContent, ProcessedContent, APIResponse
from utils.helpers import (
    ensure_directory, save_json, load_json, calculate_content_hash,
    sanitize_filename, validate_url, validate_content_quality
)
class TestHelpers:
    def test_ensure_directory(self, tmp_path):
        test_dir = tmp_path / "test_dir" / "subdir"
        ensure_directory(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()
    def test_save_and_load_json(self, tmp_path):
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        filepath = tmp_path / "test.json"
        save_json(test_data, str(filepath))
        assert filepath.exists()
        loaded_data = load_json(str(filepath))
        assert loaded_data == test_data
    def test_calculate_content_hash(self):
        content = "Test content for hashing"
        hash1 = calculate_content_hash(content)
        hash2 = calculate_content_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 32
    def test_sanitize_filename(self):
        unsafe_filename = "file<>:\"/\\|?*.txt"
        safe_filename = sanitize_filename(unsafe_filename)
        assert "<" not in safe_filename
        assert ">" not in safe_filename
        assert ":" not in safe_filename
        assert "/" not in safe_filename
        assert "\\" not in safe_filename
        assert "|" not in safe_filename
        assert "?" not in safe_filename
        assert "*" not in safe_filename
    def test_validate_url(self):
        valid_urls = [
            "https://example.com",
            "http://localhost:8000",
            "https://sub.domain.co.uk/path?param=value",
            "http://192.168.1.1:8080"
        ]
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",
            "https://",
            "http://invalid"
        ]
        for url in valid_urls:
            assert validate_url(url) == True
        for url in invalid_urls:
            assert validate_url(url) == False
    def test_validate_content_quality(self):
        result = validate_content_quality("")
        assert result["valid"] == False
        assert "Empty content" in result["reason"]
        short_content = "Short text."
        result = validate_content_quality(short_content, min_length=100)
        assert result["valid"] == False
        assert "Content too short" in result["reason"]
        long_content = "This is a longer piece of content that should pass validation. " * 10
        result = validate_content_quality(long_content, min_length=100)
        assert result["valid"] == True
        assert "Content valid" in result["reason"]
        assert "word_count" in result["metrics"]
        assert "char_count" in result["metrics"]
        assert "sentence_count" in result["metrics"]
class TestModels:
    def test_scraped_content_creation(self):
        content = ScrapedContent(
            url="https://example.com",
            title="Test Title",
            content="Test content"
        )
        assert content.url == "https://example.com"
        assert content.title == "Test Title"
        assert content.content == "Test content"
        assert content.status.value == "scraped"
        assert content.id is not None
    def test_processed_content_creation(self):
        from uuid import uuid4
        content = ProcessedContent(
            original_content_id=uuid4(),
            writer_output="Writer output",
            reviewer_output="Reviewer output",
            editor_output="Editor output",
            quality_score=0.85
        )
        assert content.quality_score == 0.85
        assert content.writer_output == "Writer output"
        assert content.reviewer_output == "Reviewer output"
        assert content.editor_output == "Editor output"
        assert content.status.value == "editing"
    def test_api_response_creation(self):
        response = APIResponse(
            success=True,
            message="Test message",
            data={"key": "value"}
        )
        assert response.success == True
        assert response.message == "Test message"
        assert response.data == {"key": "value"}
        assert response.error is None
class TestWorkflowIntegration:
    @pytest.mark.asyncio
    async def test_workflow_initialization(self):
        with patch('workflow.WebScraper') as mock_scraper:
            mock_scraper.return_value.initialize = Mock(return_value=None)
            scraper = mock_scraper()
            await scraper.initialize()
            scraper.initialize.assert_called_once()
    def test_config_loading(self):
        from config.settings import settings
        assert settings.app_name == "Book Publication Workflow"
        assert settings.api_port == 8000
        assert settings.debug == False
@pytest.mark.integration
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        pass
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 