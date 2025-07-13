import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

def ensure_directory(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: str) -> None:
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_content_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def sanitize_filename(filename: str) -> str:
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    return filename[:200] if len(filename) > 200 else filename

def format_timestamp(timestamp: datetime = None) -> str:
    if timestamp is None:
        timestamp = datetime.utcnow()
    return timestamp.strftime("%Y%m%d_%H%M%S")

def extract_domain_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except:
        return "unknown"

def truncate_text(text: str, max_length: int = 100) -> str:
    return text if len(text) <= max_length else text[:max_length-3] + "..."

def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    return max(1, len(text.split()) // words_per_minute)

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    keywords = [w for w in words if w not in stop_words and len(w) > 3]
    from collections import Counter
    return [w for w, _ in Counter(keywords).most_common(max_keywords)]

def validate_url(url: str) -> bool:
    import re
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(url))

def get_file_size_mb(filepath: str) -> float:
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / (1024 * 1024)

def create_backup_filename(original_filename: str, suffix: str = "backup") -> str:
    name, ext = os.path.splitext(original_filename)
    return f"{name}_{suffix}_{format_timestamp()}{ext}"

def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    out = dict1.copy()
    out.update(dict2)
    return out

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0):
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
                    time.sleep(delay * (2 ** attempt))
        raise last_exception
    return wrapper

def log_execution_time(func):
    import time
    import functools
    from loguru import logger
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} executed in {time.time() - start:.2f} seconds")
        return result
    return wrapper

def validate_content_quality(content: str, min_length: int = 100) -> Dict[str, Any]:
    if not content:
        return {"valid": False, "reason": "Empty content", "metrics": {}}
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