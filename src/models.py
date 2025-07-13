from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
class ContentStatus(str, Enum):
    SCRAPED = "scraped"
    WRITING = "writing"
    REVIEWING = "reviewing"
    EDITING = "editing"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
class IterationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
class FeedbackType(str, Enum):
    WRITER = "writer"
    REVIEWER = "reviewer"
    EDITOR = "editor"
    GENERAL = "general"
class VoiceCommand(str, Enum):
    START_SCRAPING = "start_scraping"
    PROCESS_CONTENT = "process_content"
    START_ITERATION = "start_iteration"
    APPROVE_CONTENT = "approve_content"
    REJECT_CONTENT = "reject_content"
    SEARCH_CONTENT = "search_content"
class ScrapedContent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    screenshot_path: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    status: ContentStatus = ContentStatus.SCRAPED
class ProcessedContent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    original_content_id: UUID
    writer_output: str
    reviewer_output: str
    editor_output: str
    quality_score: float = Field(ge=0.0, le=1.0)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    status: ContentStatus = ContentStatus.EDITING
class Iteration(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    iteration_number: int
    current_content: str
    feedback_history: List[Dict[str, Any]] = Field(default_factory=list)
    status: IterationStatus = IterationStatus.PENDING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    approved_by: Optional[str] = None
class HumanFeedback(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    iteration_id: UUID
    feedback_type: FeedbackType
    feedback_text: str
    suggested_changes: Optional[str] = None
    rating: Optional[float] = Field(None, ge=0.0, le=5.0)
    submitted_by: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
class VoiceInput(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    command: VoiceCommand
    audio_file_path: str
    transcribed_text: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
class SearchQuery(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    query_text: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    search_at: datetime = Field(default_factory=datetime.utcnow)
class RLState(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    state_vector: List[float]
    action_taken: str
    reward_received: float
    next_state: Optional[List[float]] = None
    episode_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
class WorkflowSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    session_name: str
    description: Optional[str] = None
    content_items: List[UUID] = Field(default_factory=list)
    current_iteration: Optional[UUID] = None
    status: ContentStatus = ContentStatus.SCRAPED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
class VersionControl(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content_id: UUID
    version_number: int
    content_snapshot: str
    change_description: str
    changed_by: str
    changed_at: datetime = Field(default_factory=datetime.utcnow)
    parent_version: Optional[UUID] = None
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
class ScrapingRequest(BaseModel):
    url: str
    include_screenshot: bool = True
    custom_headers: Optional[Dict[str, str]] = None
    wait_time: int = 5
class ProcessingRequest(BaseModel):
    content_id: UUID
    processing_type: str
    custom_prompts: Optional[Dict[str, str]] = None
class IterationRequest(BaseModel):
    content_id: UUID
    feedback_type: FeedbackType
    feedback_text: str
    suggested_changes: Optional[str] = None
    rating: Optional[float] = None
    user_id: str
class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    include_metadata: bool = True
class VoiceRequest(BaseModel):
    audio_file_path: str
    command_type: Optional[VoiceCommand] = None
    language: str = "en-US" 