import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def load_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def content_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def safe_filename(name):
    unsafe = '<>:"/\\|?*'
    for c in unsafe:
        name = name.replace(c, '_')
    if len(name) > 200:
        name = name[:200]
    return name

def timestamp(ts=None):
    if ts is None:
        ts = datetime.utcnow()
    return ts.strftime("%Y%m%d_%H%M%S")

def domain(url):
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc
    except:
        return "unknown"

def truncate(text, maxlen=100):
    if len(text) <= maxlen:
        return text
    return text[:maxlen-3] + "..."

def reading_time(text, wpm=200):
    count = len(text.split())
    return max(1, count // wpm)

def keywords(text, maxk=10):
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    stop = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
    }
    return [w for w in words if w not in stop][:maxk]
