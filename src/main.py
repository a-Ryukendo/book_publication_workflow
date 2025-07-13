"""
Main entry point for the Automated Book Publication Workflow
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from workflow import workflow

def run_api():
    import uvicorn
    from api.main import app
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Automated Book Publication Workflow")
    parser.add_argument("--mode", choices=["api", "cli", "example"], default="api",
                       help="Run mode: api (FastAPI server), cli (command line), example (demo)")
    parser.add_argument("--url", type=str, help="URL to process")
    parser.add_argument("--session", type=str, help="Session name")
    args = parser.parse_args()
    if args.mode == "api":
        run_api()
    elif args.mode == "cli":
        if args.url:
            asyncio.run(run_cli_workflow(args.url, args.session))
        else:
            print("Please provide a URL with --url for CLI mode")
    elif args.mode == "example":
        asyncio.run(run_example())

async def run_example():
    logger.info("Running example workflow...")
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    try:
        await workflow.initialize()
        result = await workflow.process_url(url=url, session_name="example_session", include_screenshot=True)
        if result.success:
            logger.info("Example workflow done!")
            logger.info(f"Session ID: {result.data['session_id']}")
            logger.info(f"Quality Score: {result.data['quality_score']:.3f}")
            logger.info(f"Processing Time: {result.data['processing_time']:.2f} seconds")
            cid = result.data['processed_content_id']
            iter_res = await workflow.start_iteration(cid)
            if iter_res.success:
                logger.info("Iteration started!")
                logger.info(f"Iteration ID: {iter_res.data['iteration_id']}")
            search_res = await workflow.search_content("morning gates chapter")
            if search_res.success:
                logger.info(f"Search found {len(search_res.data['results'])} results")
        else:
            logger.error(f"Example workflow failed: {result.message}")
    except Exception as e:
        logger.error(f"Example workflow error: {e}")
    finally:
        await workflow.cleanup()

async def run_cli_workflow(url: str, session: str = None):
    logger.info(f"Processing URL: {url}")
    try:
        await workflow.initialize()
        result = await workflow.process_url(url, session)
        if result.success:
            print(f"Success! Session ID: {result.data['session_id']}")
            print(f"Quality Score: {result.data['quality_score']:.3f}")
            print(f"Processing Time: {result.data['processing_time']:.2f}s")
            cid = result.data['processed_content_id']
            iter_res = await workflow.start_iteration(cid)
            if iter_res.success:
                print(f"Iteration started: {iter_res.data['iteration_id']}")
                print("Provide feedback using the API or web interface")
        else:
            print(f"Failed: {result.message}")
    except Exception as e:
        logger.error(f"CLI workflow error: {e}")
        print(f"Error: {e}")
    finally:
        await workflow.cleanup()

def main():
    if len(sys.argv) > 1:
        run_cli()
    else:
        run_api()

if __name__ == "__main__":
    main() 