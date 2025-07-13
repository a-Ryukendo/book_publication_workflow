import asyncio
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from workflow import workflow
from models import (
    ScrapedContent, ProcessedContent, Iteration, HumanFeedback,
    WorkflowSession, VersionControl, ScrapingRequest, ProcessingRequest,
    IterationRequest, SearchRequest, APIResponse
)
from config.settings import settings
from loguru import logger

app = FastAPI(
    title=settings.app_name,
    description="Automated Book Publication Workflow API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        await workflow.initialize()
        logger.info("API server started")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    try:
        await workflow.cleanup()
        logger.info("API server shutdown")
    except Exception as e:
        logger.error(f"Shutdown failed: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "1.0.0"
    }

@app.post("/api/scrape")
async def scrape_url(request: ScrapingRequest):
    try:
        result = await workflow.process_url(
            url=request.url,
            include_screenshot=request.include_screenshot
        )
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scrape/batch")
async def batch_scrape_urls(urls: list[str], include_screenshots: bool = True):
    try:
        results = [
            (await workflow.process_url(url, include_screenshot=include_screenshots)).dict()
            for url in urls
        ]
        return {
            "success": True,
            "message": f"Batch scraping: {len(results)} URLs processed",
            "data": {
                "results": results,
                "total_urls": len(urls),
                "successful": len([r for r in results if r["success"]])
            }
        }
    except Exception as e:
        logger.error(f"Batch scraping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_content(request: ProcessingRequest):
    try:
        processed = workflow.processed.get(str(request.content_id))
        if not processed:
            raise HTTPException(status_code=404, detail="Content not found")
        config = {
            "writing_style": "moderate",
            "review_focus": ["clarity", "engagement", "structure"],
            "editing_focus": ["grammar", "style", "flow"]
        }
        if request.custom_prompts:
            config.update(request.custom_prompts)
        scraped = workflow.contents.get(str(processed.original_content_id))
        if not scraped:
            raise HTTPException(status_code=404, detail="Original content not found")
        new_processed = await workflow.agent.process_content(scraped, config)
        workflow.processed[str(new_processed.id)] = new_processed
        return {
            "success": True,
            "message": "Content reprocessed",
            "data": {
                "new_processed_content_id": str(new_processed.id),
                "quality_score": new_processed.quality_score,
                "processing_metadata": new_processed.processing_metadata
            }
        }
    except Exception as e:
        logger.error(f"Content processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/iterate/start")
async def start_iteration(content_id: str, user_id: str = "default_user"):
    try:
        result = await workflow.start_iteration(content_id, user_id)
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Start iteration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/iterate/feedback")
async def submit_feedback(
    iteration_id: str,
    feedback: str = Form(...),
    feedback_type: str = Form("general"),
    user_id: str = Form("default_user"),
    suggested_changes: Optional[str] = Form(None),
    rating: Optional[float] = Form(None)
):
    try:
        result = await workflow.submit_human_feedback(
            iteration_id=iteration_id,
            feedback=feedback,
            feedback_type=feedback_type,
            user_id=user_id,
            suggested_changes=suggested_changes,
            rating=rating
        )
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/iterate/{iteration_id}")
async def get_iteration_status(iteration_id: str):
    try:
        iteration = workflow.iterations.get(iteration_id)
        if not iteration:
            raise HTTPException(status_code=404, detail="Iteration not found")
        return {
            "success": True,
            "message": "Iteration details",
            "data": {
                "iteration_id": str(iteration.id),
                "content_id": str(iteration.content_id),
                "iteration_number": iteration.iteration_number,
                "current_content": iteration.current_content,
                "status": iteration.status.value,
                "started_at": iteration.started_at.isoformat(),
                "completed_at": iteration.completed_at.isoformat() if iteration.completed_at else None,
                "approved_by": iteration.approved_by,
                "feedback_history": iteration.feedback_history
            }
        }
    except Exception as e:
        logger.error(f"Get iteration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search_content(request: SearchRequest):
    try:
        result = await workflow.search_content(
            query=request.query,
            filters=request.filters,
            limit=request.limit
        )
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/quick")
async def quick_search(query: str, limit: int = 10):
    try:
        result = await workflow.search_content(query, limit=limit)
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Quick search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/process")
async def process_voice_command(audio_file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp/{audio_file.filename}"
        os.makedirs("temp", exist_ok=True)
        with open(temp_file_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        result = await workflow.process_voice_command(temp_file_path)
        os.remove(temp_file_path)
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/speak")
async def speak_text(text: str, save_to_file: bool = False):
    try:
        audio_file_path = await workflow.voice_processor.speak_text(text, save_to_file)
        return {
            "success": True,
            "message": "Text converted to speech",
            "data": {
                "audio_file_path": audio_file_path,
                "text": text,
                "saved_to_file": save_to_file
            }
        }
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/record")
async def record_audio(duration: int = 5, sample_rate: int = 16000):
    try:
        audio_file_path = await workflow.voice_processor.record_audio(duration, sample_rate)
        return {
            "success": True,
            "message": "Audio recorded",
            "data": {
                "audio_file_path": audio_file_path,
                "duration": duration,
                "sample_rate": sample_rate
            }
        }
    except Exception as e:
        logger.error(f"Audio recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}")
async def get_session_status(session_id: str):
    try:
        result = await workflow.get_session_status(session_id)
        return JSONResponse(status_code=200 if result.success else 404, content=result.dict())
    except Exception as e:
        logger.error(f"Get session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    try:
        sessions = [
            {
                "session_id": session_id,
                "name": session.session_name,
                "status": session.status.value,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "content_count": len(session.content_items)
            }
            for session_id, session in workflow.active_sessions.items()
        ]
        return {
            "success": True,
            "message": f"Retrieved {len(sessions)} sessions",
            "data": {
                "sessions": sessions,
                "total_sessions": len(sessions)
            }
        }
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/version/create")
async def create_version(content_id: str, change_description: str, user_id: str = "default_user"):
    try:
        result = await workflow.create_version(content_id, change_description, user_id)
        return JSONResponse(status_code=200 if result.success else 400, content=result.dict())
    except Exception as e:
        logger.error(f"Create version failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/version/history/{content_id}")
async def get_version_history(content_id: str):
    try:
        # Assuming chroma_manager is available in the global scope or imported elsewhere
        # For now, we'll just return a placeholder message as chroma_manager is not defined
        # In a real scenario, this would require a proper import or initialization
        # For demonstration, we'll return a placeholder message
        return {
            "success": True,
            "message": "Version history retrieval is not yet implemented",
            "data": {
                "content_id": content_id,
                "versions": [], # Placeholder
                "total_versions": 0
            }
        }
    except Exception as e:
        logger.error(f"Get version history failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_workflow_stats():
    try:
        stats = workflow.get_workflow_stats()
        return {
            "success": True,
            "message": "Statistics retrieved",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Get stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/scraper")
async def get_scraper_stats():
    try:
        if workflow.scraper:
            stats = workflow.scraper.get_session_stats()
        else:
            stats = {}
        return {
            "success": True,
            "message": "Scraper statistics retrieved",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Get scraper stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/agents")
async def get_agent_stats():
    try:
        stats = workflow.agent_orchestrator.get_agent_stats()
        return {
            "success": True,
            "message": "Agent statistics retrieved",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Get agent stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/voice")
async def get_voice_stats():
    try:
        stats = workflow.voice_processor.get_voice_stats()
        return {
            "success": True,
            "message": "Voice statistics retrieved",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Get voice stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/chroma")
async def get_chroma_stats():
    try:
        # Assuming chroma_manager is available in the global scope or imported elsewhere
        # For now, we'll just return a placeholder message as chroma_manager is not defined
        # In a real scenario, this would require a proper import or initialization
        # For demonstration, we'll return a placeholder message
        return {
            "success": True,
            "message": "ChromaDB statistics retrieval is not yet implemented",
            "data": {} # Placeholder
        }
    except Exception as e:
        logger.error(f"Get ChromaDB stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backup")
async def backup_data():
    try:
        # Assuming chroma_manager is available in the global scope or imported elsewhere
        # For now, we'll just return a placeholder message as chroma_manager is not defined
        # In a real scenario, this would require a proper import or initialization
        # For demonstration, we'll return a placeholder message
        return {
            "success": True,
            "message": "Data backup is not yet implemented",
            "data": {} # Placeholder
        }
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/restore")
async def restore_data(backup_path: str):
    try:
        # Assuming chroma_manager is available in the global scope or imported elsewhere
        # For now, we'll just return a placeholder message as chroma_manager is not defined
        # In a real scenario, this would require a proper import or initialization
        # For demonstration, we'll return a placeholder message
        return {
            "success": True,
            "message": "Data restore is not yet implemented",
            "data": {} # Placeholder
        }
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Resource not found",
            "error": str(exc)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 