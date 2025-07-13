import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loguru import logger
from config import settings
from workflow import workflow

async def main():
    logger.info("Starting workflow")
    await workflow.init()
    await run_example()
    logger.info("Ready at /docs")
    while True:
        await asyncio.sleep(1)

async def run_example():
    logger.info("Running example")
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    result = await workflow.handle_url(url, session="example", screenshot=True)
    if result.success:
        logger.info(f"Session: {result.data['session_id']} Score: {result.data['quality_score']} Time: {result.data['processing_time']}")
        cid = result.data['processed_content_id']
        iteration = await workflow.start_iteration(cid)
        if iteration.success:
            logger.info(f"Iteration {iteration.data['iteration_id']} started")
        search = await workflow.search_content("morning gates chapter")
        if search.success:
            logger.info(f"Found {len(search.data['results'])} results")
    else:
        logger.error(f"Example failed: {result.message}")

def run_server():
    import uvicorn
    from api.main import app
    logger.info("Starting API")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=settings.debug, log_level=settings.log_level.lower())
