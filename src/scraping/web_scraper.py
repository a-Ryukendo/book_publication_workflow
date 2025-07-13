"""
Web scraping module using Playwright with RL-based optimization
"""
import asyncio
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
import json

from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import requests
from loguru import logger

from ..models import ScrapedContent, ScrapingRequest
from ..config.settings import settings
from ..rl.reward_system import ScrapingRewardSystem


class WebScraper:
    """Advanced web scraper with Playwright and RL optimization"""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.reward_system = ScrapingRewardSystem()
        self.session_stats = {
            "successful_scrapes": 0,
            "failed_scrapes": 0,
            "total_content_length": 0,
            "average_quality_score": 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize Playwright browser"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu'
                ]
            )
            logger.info("Web scraper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize web scraper: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        logger.info("Web scraper cleaned up")
    
    async def scrape_url(self, request: ScrapingRequest) -> ScrapedContent:
        """Scrape content from URL with RL optimization"""
        start_time = time.time()
        
        try:
            # Create new page
            page = await self.browser.new_page()
            
            # Set viewport and user agent
            await page.set_viewport_size({"width": 1920, "height": 1080})
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            # Navigate to URL with timeout
            logger.info(f"Navigating to: {request.url}")
            await page.goto(request.url, timeout=settings.scraping_timeout * 1000)
            
            # Wait for page to load
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(request.wait_time)
            
            # Extract content
            content_data = await self._extract_content(page)
            
            # Take screenshot if requested
            screenshot_path = None
            if request.include_screenshot:
                screenshot_path = await self._take_screenshot(page, request.url)
            
            # Calculate quality score using RL
            quality_score = self.reward_system.calculate_quality_score(
                content_data["content"],
                content_data["title"],
                len(content_data["content"])
            )
            
            # Update RL state
            state_vector = self._create_state_vector(content_data, quality_score)
            self.reward_system.update_state(state_vector, "scrape", quality_score)
            
            # Create scraped content object
            scraped_content = ScrapedContent(
                url=request.url,
                title=content_data["title"],
                content=content_data["content"],
                metadata={
                    "quality_score": quality_score,
                    "word_count": len(content_data["content"].split()),
                    "scraping_time": time.time() - start_time,
                    "screenshot_taken": request.include_screenshot,
                    "extracted_metadata": content_data["metadata"]
                },
                screenshot_path=screenshot_path,
                status="scraped"
            )
            
            # Update session stats
            self._update_session_stats(quality_score, len(content_data["content"]))
            
            logger.info(f"Successfully scraped {request.url} with quality score: {quality_score:.3f}")
            return scraped_content
            
        except Exception as e:
            logger.error(f"Failed to scrape {request.url}: {e}")
            # Update RL with negative reward
            self.reward_system.update_state([0] * 10, "scrape", -1.0)
            self.session_stats["failed_scrapes"] += 1
            raise
        finally:
            if 'page' in locals():
                await page.close()
    
    async def _extract_content(self, page: Page) -> Dict[str, Any]:
        """Extract content from page using multiple strategies"""
        content_data = {
            "title": "",
            "content": "",
            "metadata": {}
        }
        
        # Extract title
        title_selectors = [
            "h1",
            "title",
            "[class*='title']",
            "[id*='title']"
        ]
        
        for selector in title_selectors:
            try:
                title_element = await page.query_selector(selector)
                if title_element:
                    title_text = await title_element.text_content()
                    if title_text and len(title_text.strip()) > 0:
                        content_data["title"] = title_text.strip()
                        break
            except:
                continue
        
        # Extract main content
        content_selectors = [
            "main",
            "article",
            "[class*='content']",
            "[id*='content']",
            "[class*='text']",
            "[id*='text']",
            "body"
        ]
        
        for selector in content_selectors:
            try:
                content_element = await page.query_selector(selector)
                if content_element:
                    content_text = await content_element.text_content()
                    if content_text and len(content_text.strip()) > 100:
                        content_data["content"] = content_text.strip()
                        break
            except:
                continue
        
        # Extract metadata
        content_data["metadata"] = await self._extract_metadata(page)
        
        return content_data
    
    async def _extract_metadata(self, page: Page) -> Dict[str, Any]:
        """Extract metadata from page"""
        metadata = {}
        
        try:
            # Extract meta tags
            meta_tags = await page.query_selector_all("meta")
            for tag in meta_tags:
                name = await tag.get_attribute("name") or await tag.get_attribute("property")
                content = await tag.get_attribute("content")
                if name and content:
                    metadata[name] = content
            
            # Extract structured data
            script_tags = await page.query_selector_all("script[type='application/ld+json']")
            for script in script_tags:
                try:
                    json_content = await script.text_content()
                    if json_content:
                        structured_data = json.loads(json_content)
                        metadata["structured_data"] = structured_data
                except:
                    continue
            
            # Extract links
            links = await page.query_selector_all("a[href]")
            metadata["link_count"] = len(links)
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    async def _take_screenshot(self, page: Page, url: str) -> str:
        """Take screenshot of the page"""
        try:
            # Create filename based on URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace(".", "_")
            path = parsed_url.path.replace("/", "_").replace(".", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}{path}_{timestamp}.png"
            
            # Ensure screenshots directory exists
            os.makedirs(settings.screenshots_dir, exist_ok=True)
            screenshot_path = os.path.join(settings.screenshots_dir, filename)
            
            # Take screenshot
            await page.screenshot(
                path=screenshot_path,
                full_page=True,
                quality=settings.screenshot_quality
            )
            
            logger.info(f"Screenshot saved: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
    
    def _create_state_vector(self, content_data: Dict[str, Any], quality_score: float) -> list:
        """Create state vector for RL"""
        return [
            len(content_data["content"]),  # Content length
            len(content_data["title"]),    # Title length
            quality_score,                 # Quality score
            content_data["metadata"].get("link_count", 0),  # Link count
            len(content_data["content"].split()),  # Word count
            float(bool(content_data["title"])),    # Has title
            float(len(content_data["content"]) > 100),  # Has substantial content
            float(quality_score > 0.5),    # Good quality
            float(len(content_data["metadata"]) > 0),  # Has metadata
            time.time() % 1000  # Time component
        ]
    
    def _update_session_stats(self, quality_score: float, content_length: int):
        """Update session statistics"""
        self.session_stats["successful_scrapes"] += 1
        self.session_stats["total_content_length"] += content_length
        
        # Update average quality score
        total_scrapes = self.session_stats["successful_scrapes"]
        current_avg = self.session_stats["average_quality_score"]
        self.session_stats["average_quality_score"] = (
            (current_avg * (total_scrapes - 1) + quality_score) / total_scrapes
        )
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return self.session_stats.copy()
    
    async def batch_scrape(self, urls: list, include_screenshots: bool = True) -> list:
        """Scrape multiple URLs in parallel"""
        tasks = []
        for url in urls:
            request = ScrapingRequest(url=url, include_screenshot=include_screenshots)
            tasks.append(self.scrape_url(request))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch scrape failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results


# Utility function for synchronous usage
def scrape_url_sync(url: str, include_screenshot: bool = True) -> ScrapedContent:
    """Synchronous wrapper for scraping"""
    async def _scrape():
        async with WebScraper() as scraper:
            request = ScrapingRequest(url=url, include_screenshot=include_screenshot)
            return await scraper.scrape_url(request)
    
    return asyncio.run(_scrape()) 