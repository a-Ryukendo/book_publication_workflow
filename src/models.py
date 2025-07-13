from dataclasses import dataclass, field

@dataclass
class Scraped:
    id: str
    title: str
    body: str
    info: dict = field(default_factory=dict)
    screenshot: str = ""
    created: str = ""
    status: str = "scraped"

@dataclass
class Processed:
    original_id: str
    writer: str
    reviewer: str
    editor: str
    score: float
    meta: dict = field(default_factory=dict)
    finished: str = ""
    status: str = "editing"

@dataclass
class Iteration:
    id: str
    content_id: str
    round: int
    text: str
    feedback: list = field(default_factory=list)
    status: str = "pending"
    started: str = ""
    ended: str = ""
    approved_by: str = ""

@dataclass
class Feedback:
    id: str
    iteration_id: str
    type: str
    message: str
    suggestion: str = ""
    rating: float = None
    user: str = ""
    submitted: str = ""

@dataclass
class Voice:
    id: str
    command: str
    audio_path: str
    transcript: str
