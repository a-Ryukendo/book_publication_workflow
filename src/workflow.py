import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from loguru import logger
from models import (
    ScrapedContent, ProcessedContent, Iteration, HumanFeedback,
    WorkflowSession, VersionControl, ScrapingRequest, ProcessingRequest,
    IterationRequest, SearchRequest, APIResponse
)
from scraping.web_scraper import WebScraper
from agents.ai_agents import AgentOrchestrator
from database.chroma_manager import chroma_manager
from voice.voice_processor import VoiceProcessor, VoiceCommandProcessor
from config.settings import settings

class BookPublicationWorkflow:
    def __init__(self):
        self.scraper = None
        self.agent = AgentOrchestrator()
        self.voice = VoiceProcessor()
        self.voice_cmd = VoiceCommandProcessor(self)
        self.sessions: Dict[str, WorkflowSession] = {}
        self.contents: Dict[str, ScrapedContent] = {}
        self.processed: Dict[str, ProcessedContent] = {}
        self.iterations: Dict[str, Iteration] = {}
        self.stats = {
            "total_sessions": 0,
            "total_content": 0,
            "total_iterations": 0,
            "avg_time": 0.0,
            "success_rate": 0.0
        }

    async def initialize(self):
        try:
            self.scraper = WebScraper()
            await self.scraper.initialize()
            logger.info("Workflow ready")
        except Exception as e:
            logger.error(f"Init failed: {e}")
            raise

    async def cleanup(self):
        try:
            if self.scraper:
                await self.scraper.cleanup()
            self.voice.cleanup()
            self.voice_cmd.cleanup()
            self.agent.save_models()
            logger.info("Workflow cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    async def process_url(self, url: str, session_name: str = None, include_screenshot: bool = True) -> APIResponse:
        start = time.time()
        try:
            session = await self._get_or_create_session(session_name or f"session_{uuid4().hex[:8]}")
            logger.info(f"Scraping: {url}")
            req = ScrapingRequest(url=url, include_screenshot=include_screenshot)
            scraped = await self.scraper.scrape_url(req)
            self.contents[str(scraped.id)] = scraped
            chroma_manager.add_scraped_content(scraped)
            session.content_items.append(scraped.id)
            session.status = "scraped"
            session.updated_at = datetime.utcnow()
            logger.info("AI processing...")
            config = {
                "writing_style": "moderate",
                "review_focus": ["clarity", "engagement", "structure"],
                "editing_focus": ["grammar", "style", "flow"]
            }
            processed = await self.agent.process_content(scraped, config)
            self.processed[str(processed.id)] = processed
            chroma_manager.add_processed_content(processed, scraped)
            session.status = "processed"
            session.updated_at = datetime.utcnow()
            elapsed = time.time() - start
            self._update_stats(True, elapsed)
            logger.info(f"Done in {elapsed:.2f}s")
            return APIResponse(
                success=True,
                message="Content processed",
                data={
                    "session_id": str(session.id),
                    "scraped_content_id": str(scraped.id),
                    "processed_content_id": str(processed.id),
                    "processing_time": elapsed,
                    "quality_score": processed.quality_score
                }
            )
        except Exception as e:
            logger.error(f"Process failed: {e}")
            self._update_stats(False, time.time() - start)
            return APIResponse(success=False, message=f"Failed: {str(e)}", error=str(e))

    async def start_iteration(self, content_id: str, user_id: str = "default_user") -> APIResponse:
        try:
            processed = self.processed.get(content_id)
            if not processed:
                return APIResponse(success=False, message="Processed content not found", error="Content ID not found")
            iteration = Iteration(
                content_id=UUID(content_id),
                iteration_number=1,
                current_content=processed.editor_output,
                status="pending"
            )
            self.iterations[str(iteration.id)] = iteration
            session = await self._find_session_by_content(content_id)
            if session:
                session.current_iteration = iteration.id
                session.status = "iterating"
                session.updated_at = datetime.utcnow()
            logger.info(f"Started iteration for {content_id}")
            return APIResponse(
                success=True,
                message="Iteration started",
                data={
                    "iteration_id": str(iteration.id),
                    "content_id": content_id,
                    "current_content": iteration.current_content,
                    "status": iteration.status.value
                }
            )
        except Exception as e:
            logger.error(f"Iteration start failed: {e}")
            return APIResponse(success=False, message=f"Iteration start failed: {str(e)}", error=str(e))

    async def submit_human_feedback(self, iteration_id: str, feedback: str, feedback_type: str = "general", user_id: str = "default_user", suggested_changes: str = None, rating: float = None) -> APIResponse:
        try:
            iteration = self.iterations.get(iteration_id)
            if not iteration:
                return APIResponse(success=False, message="Iteration not found", error="Iteration ID not found")
            human_feedback = HumanFeedback(
                iteration_id=UUID(iteration_id),
                feedback_type=feedback_type,
                feedback_text=feedback,
                suggested_changes=suggested_changes,
                rating=rating,
                submitted_by=user_id
            )
            iteration.feedback_history.append({
                "feedback_id": str(human_feedback.id),
                "feedback_type": feedback_type,
                "feedback_text": feedback,
                "suggested_changes": suggested_changes,
                "rating": rating,
                "submitted_by": user_id,
                "submitted_at": human_feedback.submitted_at.isoformat()
            })
            if suggested_changes:
                iteration.current_content = suggested_changes
                iteration.status = "in_progress"
            else:
                await self._process_feedback_with_ai(iteration, feedback, feedback_type)
            if rating and rating >= settings.approval_threshold:
                iteration.status = "approved"
                iteration.approved_by = user_id
                iteration.completed_at = datetime.utcnow()
                session = await self._find_session_by_iteration(iteration_id)
                if session:
                    session.status = "approved"
                    session.updated_at = datetime.utcnow()
            logger.info(f"Feedback submitted for {iteration_id}")
            return APIResponse(
                success=True,
                message="Feedback submitted",
                data={
                    "iteration_id": iteration_id,
                    "feedback_id": str(human_feedback.id),
                    "iteration_status": iteration.status.value,
                    "current_content": iteration.current_content
                }
            )
        except Exception as e:
            logger.error(f"Feedback failed: {e}")
            return APIResponse(success=False, message=f"Feedback failed: {str(e)}", error=str(e))

    async def _process_feedback_with_ai(self, iteration: Iteration, feedback: str, feedback_type: str):
        try:
            if feedback_type == "writer":
                processed = self.processed.get(str(iteration.content_id))
                if processed:
                    temp_scraped = ScrapedContent(
                        url="",
                        title="Feedback Processing",
                        content=iteration.current_content
                    )
                    updated = await self.agent.writer.write_content(
                        temp_scraped,
                        style="adaptive",
                        custom_prompt=f"Address this feedback: {feedback}"
                    )
                    iteration.current_content = updated
            elif feedback_type == "reviewer":
                review = await self.agent.reviewer.review_content(
                    iteration.current_content,
                    focus_areas=["clarity", "engagement", "structure"]
                )
                if review.get("recommendations"):
                    updated = await self.agent.editor.edit_content(
                        iteration.current_content,
                        review_feedback=review,
                        focus_areas=["clarity", "engagement", "structure"]
                    )
                    iteration.current_content = updated
            iteration.status = "in_progress"
        except Exception as e:
            logger.error(f"AI feedback failed: {e}")

    async def search_content(self, query: str, filters: Dict[str, Any] = None, limit: int = 10) -> APIResponse:
        try:
            results = chroma_manager.semantic_search(query, filters, limit)
            return APIResponse(
                success=True,
                message=f"Search: {len(results)} results",
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                }
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return APIResponse(success=False, message=f"Search failed: {str(e)}", error=str(e))

    async def get_session_status(self, session_id: str) -> APIResponse:
        try:
            session = self.sessions.get(session_id)
            if not session:
                return APIResponse(success=False, message="Session not found", error="Session ID not found")
            content_details = []
            for cid in session.content_items:
                scraped = self.contents.get(str(cid))
                processed = self.processed.get(str(cid))
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
                message="Session status",
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
            logger.error(f"Session status failed: {e}")
            return APIResponse(success=False, message=f"Session status failed: {str(e)}", error=str(e))

    async def create_version(self, content_id: str, change_description: str, user_id: str = "default_user") -> APIResponse:
        try:
            processed = self.processed.get(content_id)
            if not processed:
                return APIResponse(success=False, message="Content not found", error="Content ID not found")
            version_history = chroma_manager.get_version_history(content_id)
            version_number = len(version_history) + 1
            version = VersionControl(
                content_id=UUID(content_id),
                version_number=version_number,
                content_snapshot=processed.editor_output,
                change_description=change_description,
                changed_by=user_id
            )
            chroma_manager.add_version_control(version)
            logger.info(f"Version {version_number} for {content_id}")
            return APIResponse(
                success=True,
                message=f"Version {version_number} created",
                data={
                    "version_id": str(version.id),
                    "version_number": version_number,
                    "content_id": content_id,
                    "change_description": change_description
                }
            )
        except Exception as e:
            logger.error(f"Version failed: {e}")
            return APIResponse(success=False, message=f"Version failed: {str(e)}", error=str(e))

    async def _get_or_create_session(self, session_name: str) -> WorkflowSession:
        for session in self.sessions.values():
            if session.session_name == session_name:
                return session
        session = WorkflowSession(
            session_name=session_name,
            description=f"Session for {session_name}"
        )
        self.sessions[str(session.id)] = session
        self.stats["total_sessions"] += 1
        return session

    async def _find_session_by_content(self, content_id: str) -> Optional[WorkflowSession]:
        for session in self.sessions.values():
            if any(str(cid) == content_id for cid in session.content_items):
                return session
        return None

    async def _find_session_by_iteration(self, iteration_id: str) -> Optional[WorkflowSession]:
        for session in self.sessions.values():
            if session.current_iteration and str(session.current_iteration) == iteration_id:
                return session
        return None

    def _update_stats(self, success: bool, elapsed: float):
        self.stats["total_content"] += 1
        total = self.stats["total_content"]
        avg = self.stats["avg_time"]
        self.stats["avg_time"] = ((avg * (total - 1) + elapsed) / total)
        if success:
            self.stats["success_rate"] = ((self.stats["success_rate"] * (total - 1) + 1) / total)
        else:
            self.stats["success_rate"] = ((self.stats["success_rate"] * (total - 1)) / total)

    def get_workflow_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats.update({
            "agent_stats": self.agent.get_agent_stats(),
            "voice_stats": self.voice.get_voice_stats(),
            "chroma_stats": chroma_manager.get_collection_stats(),
            "scraper_stats": self.scraper.get_session_stats() if self.scraper else {}
        })
        return stats

    async def process_voice_command(self, audio_file_path: str) -> APIResponse:
        try:
            voice_req = VoiceRequest(audio_file_path=audio_file_path)
            voice_input = await self.voice.process_voice_input(voice_req)
            result = await self.voice_cmd.process_voice_command(voice_input)
            return APIResponse(
                success=result["success"],
                message=result["message"],
                data=result["data"]
            )
        except Exception as e:
            logger.error(f"Voice command failed: {e}")
            return APIResponse(success=False, message=f"Voice command failed: {str(e)}", error=str(e))

workflow = BookPublicationWorkflow() 