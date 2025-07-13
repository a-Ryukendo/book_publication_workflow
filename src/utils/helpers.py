"""
Helper utility functions for the Automated Book Publication Workflow
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file"""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_content_hash(content: str) -> str:
    """Calculate MD5 hash of content"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage"""
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def format_timestamp(timestamp: datetime = None) -> str:
    """Format timestamp for file naming"""
    if timestamp is None:
        timestamp = datetime.utcnow()
    return timestamp.strftime("%Y%m%d_%H%M%S")


def extract_domain_from_url(url: str) -> str:
    """Extract domain from URL"""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return "unknown"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Calculate estimated reading time in minutes"""
    word_count = len(text.split())
    return max(1, word_count // words_per_minute)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    # Simple keyword extraction - in production, use NLP libraries
    import re
    
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words
    words = text.split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Count frequency
    from collections import Counter
    word_counts = Counter(keywords)
    
    # Return most common keywords
    return [word for word, count in word_counts.most_common(max_keywords)]


def validate_url(url: str) -> bool:
    """Validate URL format"""
    import re
    
    # Simple URL validation pattern
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return bool(url_pattern.match(url))


def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes"""
    if not os.path.exists(filepath):
        return 0.0
    
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)


def create_backup_filename(original_filename: str, suffix: str = "backup") -> str:
    """Create backup filename with timestamp"""
    name, ext = os.path.splitext(original_filename)
    timestamp = format_timestamp()
    return f"{name}_{suffix}_{timestamp}{ext}"


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with dict2 taking precedence"""
    result = dict1.copy()
    result.update(dict2)
    return result


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on exception"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
        
        raise last_exception
    
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    import functools
    from loguru import logger
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        
        return result
    
    return wrapper


def validate_content_quality(content: str, min_length: int = 100) -> Dict[str, Any]:
    """Validate content quality and return metrics"""
    if not content:
        return {
            "valid": False,
            "reason": "Empty content",
            "metrics": {}
        }
    
    word_count = len(content.split())
    char_count = len(content)
    sentence_count = content.count('.') + content.count('!') + content.count('?')
    paragraph_count = content.count('\n\n') + 1
    
    metrics = {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "avg_sentence_length": word_count / max(sentence_count, 1),
        "avg_paragraph_length": word_count / paragraph_count
    }
    
    valid = char_count >= min_length
    
    return {
        "valid": valid,
        "reason": "Content too short" if not valid else "Content valid",
        "metrics": metrics
    } 