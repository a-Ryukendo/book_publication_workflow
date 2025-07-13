"""
Main workflow orchestrator for the Automated Book Publication Workflow
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from loguru import logger

from .models import (
    ScrapedContent, ProcessedContent, Iteration, HumanFeedback,
    WorkflowSession, VersionControl, ScrapingRequest, ProcessingRequest,
    IterationRequest, SearchRequest, APIResponse
)
from .scraping.web_scraper import WebScraper
from .agents.ai_agents import AgentOrchestrator
from .database.chroma_manager import chroma_manager
from .voice.voice_processor import VoiceProcessor, VoiceCommandProcessor
from .config.settings import settings


class BookPublicationWorkflow:
    """Main orchestrator for the book publication workflow"""
    
    def __init__(self):
        self.scraper = None
        self.agent_orchestrator = AgentOrchestrator()
        self.voice_processor = VoiceProcessor()
        self.voice_command_processor = VoiceCommandProcessor(self)
        
        # Session management
        self.active_sessions: Dict[str, WorkflowSession] = {}
        self.content_store: Dict[str, ScrapedContent] = {}
        self.processed_content_store: Dict[str, ProcessedContent] = {}
        self.iterations_store: Dict[str, Iteration] = {}
        
        # Performance tracking
        self.workflow_stats = {
            "total_sessions": 0,
            "total_content_items": 0,
            "total_iterations": 0,
            "average_processing_time": 0.0,
            "success_rate": 0.0
        }
    
    async def initialize(self):
        """Initialize the workflow system"""
        try:
            # Initialize web scraper
            self.scraper = WebScraper()
            await self.scraper.initialize()
            
            logger.info("Book Publication Workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.scraper:
                await self.scraper.cleanup()
            
            self.voice_processor.cleanup()
            self.voice_command_processor.cleanup()
            
            # Save RL models
            self.agent_orchestrator.save_models()
            
            logger.info("Workflow cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def process_url(self, url: str, session_name: str = None, 
                         include_screenshot: bool = True) -> APIResponse:
        """Process content from URL through the full workflow"""
        start_time = time.time()
        
        try:
            # Create or get session
            session = await self._get_or_create_session(session_name or f"session_{uuid4().hex[:8]}")
            
            # Step 1: Scrape content
            logger.info(f"Starting scraping for URL: {url}")
            scraping_request = ScrapingRequest(
                url=url,
                include_screenshot=include_screenshot
            )
            
            scraped_content = await self.scraper.scrape_url(scraping_request)
            self.content_store[str(scraped_content.id)] = scraped_content
            
            # Add to ChromaDB
            chroma_manager.add_scraped_content(scraped_content)
            
            # Update session
            session.content_items.append(scraped_content.id)
            session.status = "scraped"
            session.updated_at = datetime.utcnow()
            
            # Step 2: Process with AI agents
            logger.info("Starting AI content processing")
            processing_config = {
                "writing_style": "moderate",
                "review_focus": ["clarity", "engagement", "structure"],
                "editing_focus": ["grammar", "style", "flow"]
            }
            
            processed_content = await self.agent_orchestrator.process_content(
                scraped_content, processing_config
            )
            self.processed_content_store[str(processed_content.id)] = processed_content
            
            # Add to ChromaDB
            chroma_manager.add_processed_content(processed_content, scraped_content)
            
            # Update session
            session.status = "processed"
            session.updated_at = datetime.utcnow()
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_workflow_stats(True, processing_time)
            
            logger.info(f"URL processing completed in {processing_time:.2f} seconds")
            
            return APIResponse(
                success=True,
                message="Content processed successfully",
                data={
                    "session_id": str(session.id),
                    "scraped_content_id": str(scraped_content.id),
                    "processed_content_id": str(processed_content.id),
                    "processing_time": processing_time,
                    "quality_score": processed_content.quality_score
                }
            )
            
        except Exception as e:
            logger.error(f"URL processing failed: {e}")
            self._update_workflow_stats(False, time.time() - start_time)
            
            return APIResponse(
                success=False,
                message=f"Processing failed: {str(e)}",
                error=str(e)
            )
    
    async def start_iteration(self, content_id: str, user_id: str = "default_user") -> APIResponse:
        """Start human-in-the-loop iteration"""
        try:
            # Get processed content
            processed_content = self.processed_content_store.get(content_id)
            if not processed_content:
                return APIResponse(
                    success=False,
                    message="Processed content not found",
                    error="Content ID not found"
                )
            
            # Create iteration
            iteration = Iteration(
                content_id=UUID(content_id),
                iteration_number=1,
                current_content=processed_content.editor_output,
                status="pending"
            )
            
            self.iterations_store[str(iteration.id)] = iteration
            
            # Update session
            session = await self._find_session_by_content(content_id)
            if session:
                session.current_iteration = iteration.id
                session.status = "iterating"
                session.updated_at = datetime.utcnow()
            
            logger.info(f"Started iteration for content: {content_id}")
            
            return APIResponse(
                success=True,
                message="Iteration started successfully",
                data={
                    "iteration_id": str(iteration.id),
                    "content_id": content_id,
                    "current_content": iteration.current_content,
                    "status": iteration.status.value
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to start iteration: {e}")
            return APIResponse(
                success=False,
                message=f"Iteration start failed: {str(e)}",
                error=str(e)
            )
    
    async def submit_human_feedback(self, iteration_id: str, feedback: str,
                                  feedback_type: str = "general", user_id: str = "default_user",
                                  suggested_changes: str = None, rating: float = None) -> APIResponse:
        """Submit human feedback for iteration"""
        try:
            # Get iteration
            iteration = self.iterations_store.get(iteration_id)
            if not iteration:
                return APIResponse(
                    success=False,
                    message="Iteration not found",
                    error="Iteration ID not found"
                )
            
            # Create feedback object
            human_feedback = HumanFeedback(
                iteration_id=UUID(iteration_id),
                feedback_type=feedback_type,
                feedback_text=feedback,
                suggested_changes=suggested_changes,
                rating=rating,
                submitted_by=user_id
            )
            
            # Add feedback to iteration
            iteration.feedback_history.append({
                "feedback_id": str(human_feedback.id),
                "feedback_type": feedback_type,
                "feedback_text": feedback,
                "suggested_changes": suggested_changes,
                "rating": rating,
                "submitted_by": user_id,
                "submitted_at": human_feedback.submitted_at.isoformat()
            })
            
            # Process feedback with AI agents
            if suggested_changes:
                # Apply suggested changes
                iteration.current_content = suggested_changes
                iteration.status = "in_progress"
            else:
                # Use AI to process feedback
                await self._process_feedback_with_ai(iteration, feedback, feedback_type)
            
            # Check if iteration should be completed
            if rating and rating >= settings.approval_threshold:
                iteration.status = "approved"
                iteration.approved_by = user_id
                iteration.completed_at = datetime.utcnow()
                
                # Update session
                session = await self._find_session_by_iteration(iteration_id)
                if session:
                    session.status = "approved"
                    session.updated_at = datetime.utcnow()
            
            logger.info(f"Human feedback submitted for iteration: {iteration_id}")
            
            return APIResponse(
                success=True,
                message="Feedback submitted successfully",
                data={
                    "iteration_id": iteration_id,
                    "feedback_id": str(human_feedback.id),
                    "iteration_status": iteration.status.value,
                    "current_content": iteration.current_content
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return APIResponse(
                success=False,
                message=f"Feedback submission failed: {str(e)}",
                error=str(e)
            )
    
    async def _process_feedback_with_ai(self, iteration: Iteration, feedback: str, feedback_type: str):
        """Process human feedback with AI agents"""
        try:
            # Create processing request based on feedback type
            if feedback_type == "writer":
                # Use AI Writer to address feedback
                processed_content = self.processed_content_store.get(str(iteration.content_id))
                if processed_content:
                    # Create temporary scraped content for processing
                    temp_scraped = ScrapedContent(
                        url="",
                        title="Feedback Processing",
                        content=iteration.current_content
                    )
                    
                    # Process with AI Writer
                    updated_content = await self.agent_orchestrator.writer.write_content(
                        temp_scraped,
                        style="adaptive",
                        custom_prompt=f"Address this feedback: {feedback}"
                    )
                    
                    iteration.current_content = updated_content
            
            elif feedback_type == "reviewer":
                # Use AI Reviewer to assess feedback
                review_result = await self.agent_orchestrator.reviewer.review_content(
                    iteration.current_content,
                    focus_areas=["clarity", "engagement", "structure"]
                )
                
                # Apply reviewer suggestions
                if review_result.get("recommendations"):
                    # Use AI Editor to apply recommendations
                    updated_content = await self.agent_orchestrator.editor.edit_content(
                        iteration.current_content,
                        review_feedback=review_result,
                        focus_areas=["clarity", "engagement", "structure"]
                    )
                    
                    iteration.current_content = updated_content
            
            iteration.status = "in_progress"
            
        except Exception as e:
            logger.error(f"AI feedback processing failed: {e}")
    
    async def search_content(self, query: str, filters: Dict[str, Any] = None, 
                           limit: int = 10) -> APIResponse:
        """Search content using semantic search"""
        try:
            results = chroma_manager.semantic_search(query, filters, limit)
            
            return APIResponse(
                success=True,
                message=f"Search completed: {len(results)} results found",
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                }
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return APIResponse(
                success=False,
                message=f"Search failed: {str(e)}",
                error=str(e)
            )
    
    async def get_session_status(self, session_id: str) -> APIResponse:
        """Get session status and details"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return APIResponse(
                    success=False,
                    message="Session not found",
                    error="Session ID not found"
                )
            
            # Get content details
            content_details = []
            for content_id in session.content_items:
                scraped = self.content_store.get(str(content_id))
                processed = self.processed_content_store.get(str(content_id))
                
                if scraped and processed:
                    content_details.append({
                        "scraped_content": {
                            "id": str(scraped.id),
                            "title": scraped.title,
                            "url": scraped.url,
                            "status": scraped.status.value
                        },
                        "processed_content": {
                            "id": str(processed.id),
                            "quality_score": processed.quality_score,
                            "status": processed.status.value
                        }
                    })
            
            return APIResponse(
                success=True,
                message="Session status retrieved",
                data={
                    "session": {
                        "id": str(session.id),
                        "name": session.session_name,
                        "status": session.status.value,
                        "created_at": session.created_at.isoformat(),
                        "updated_at": session.updated_at.isoformat()
                    },
                    "content_items": content_details,
                    "current_iteration": str(session.current_iteration) if session.current_iteration else None
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get session status: {e}")
            return APIResponse(
                success=False,
                message=f"Failed to get session status: {str(e)}",
                error=str(e)
            )
    
    async def create_version(self, content_id: str, change_description: str, 
                           user_id: str = "default_user") -> APIResponse:
        """Create a new version of content"""
        try:
            # Get current content
            processed_content = self.processed_content_store.get(content_id)
            if not processed_content:
                return APIResponse(
                    success=False,
                    message="Content not found",
                    error="Content ID not found"
                )
            
            # Get version number
            version_history = chroma_manager.get_version_history(content_id)
            version_number = len(version_history) + 1
            
            # Create version control entry
            version = VersionControl(
                content_id=UUID(content_id),
                version_number=version_number,
                content_snapshot=processed_content.editor_output,
                change_description=change_description,
                changed_by=user_id
            )
            
            # Add to ChromaDB
            chroma_manager.add_version_control(version)
            
            logger.info(f"Created version {version_number} for content: {content_id}")
            
            return APIResponse(
                success=True,
                message=f"Version {version_number} created successfully",
                data={
                    "version_id": str(version.id),
                    "version_number": version_number,
                    "content_id": content_id,
                    "change_description": change_description
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            return APIResponse(
                success=False,
                message=f"Version creation failed: {str(e)}",
                error=str(e)
            )
    
    async def _get_or_create_session(self, session_name: str) -> WorkflowSession:
        """Get or create a workflow session"""
        # Check if session exists
        for session in self.active_sessions.values():
            if session.session_name == session_name:
                return session
        
        # Create new session
        session = WorkflowSession(
            session_name=session_name,
            description=f"Session for {session_name}"
        )
        
        self.active_sessions[str(session.id)] = session
        self.workflow_stats["total_sessions"] += 1
        
        return session
    
    async def _find_session_by_content(self, content_id: str) -> Optional[WorkflowSession]:
        """Find session by content ID"""
        for session in self.active_sessions.values():
            if any(str(cid) == content_id for cid in session.content_items):
                return session
        return None
    
    async def _find_session_by_iteration(self, iteration_id: str) -> Optional[WorkflowSession]:
        """Find session by iteration ID"""
        for session in self.active_sessions.values():
            if session.current_iteration and str(session.current_iteration) == iteration_id:
                return session
        return None
    
    def _update_workflow_stats(self, success: bool, processing_time: float):
        """Update workflow statistics"""
        self.workflow_stats["total_content_items"] += 1
        
        # Update average processing time
        total_items = self.workflow_stats["total_content_items"]
        current_avg = self.workflow_stats["average_processing_time"]
        self.workflow_stats["average_processing_time"] = (
            (current_avg * (total_items - 1) + processing_time) / total_items
        )
        
        # Update success rate
        if success:
            self.workflow_stats["success_rate"] = (
                (self.workflow_stats["success_rate"] * (total_items - 1) + 1) / total_items
            )
        else:
            self.workflow_stats["success_rate"] = (
                (self.workflow_stats["success_rate"] * (total_items - 1)) / total_items
            )
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        stats = self.workflow_stats.copy()
        
        # Add component stats
        stats.update({
            "agent_stats": self.agent_orchestrator.get_agent_stats(),
            "voice_stats": self.voice_processor.get_voice_stats(),
            "chroma_stats": chroma_manager.get_collection_stats(),
            "scraper_stats": self.scraper.get_session_stats() if self.scraper else {}
        })
        
        return stats
    
    async def process_voice_command(self, audio_file_path: str) -> APIResponse:
        """Process voice command"""
        try:
            # Process voice input
            voice_request = VoiceRequest(audio_file_path=audio_file_path)
            voice_input = await self.voice_processor.process_voice_input(voice_request)
            
            # Process command
            result = await self.voice_command_processor.process_voice_command(voice_input)
            
            return APIResponse(
                success=result["success"],
                message=result["message"],
                data=result["data"]
            )
            
        except Exception as e:
            logger.error(f"Voice command processing failed: {e}")
            return APIResponse(
                success=False,
                message=f"Voice command failed: {str(e)}",
                error=str(e)
            )


# Global workflow instance
workflow = BookPublicationWorkflow() 