"""
Main entry point for the Automated Book Publication Workflow
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the src directory and parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from workflow import workflow


async def main():
    """Main function to run the workflow system"""
    try:
        logger.info("Starting Automated Book Publication Workflow")
        
        # Initialize the workflow
        await workflow.initialize()
        
        # Example usage
        await run_example_workflow()
        
        # Keep the system running for API access
        logger.info("System initialized. Use the API endpoints to interact with the workflow.")
        logger.info(f"API documentation available at: http://{settings.api_host}:{settings.api_port}/docs")
        
        # Wait indefinitely (in a real application, you'd run the FastAPI server here)
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await workflow.cleanup()


async def run_example_workflow():
    """Run an example workflow to demonstrate the system"""
    try:
        logger.info("Running example workflow...")
        
        # Example URL from the requirements
        example_url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
        
        # Process the URL
        result = await workflow.process_url(
            url=example_url,
            session_name="example_session",
            include_screenshot=True
        )
        
        if result.success:
            logger.info("Example workflow completed successfully!")
            logger.info(f"Session ID: {result.data['session_id']}")
            logger.info(f"Quality Score: {result.data['quality_score']:.3f}")
            logger.info(f"Processing Time: {result.data['processing_time']:.2f} seconds")
            
            # Start an iteration
            content_id = result.data['processed_content_id']
            iteration_result = await workflow.start_iteration(content_id)
            
            if iteration_result.success:
                logger.info("Iteration started successfully!")
                logger.info(f"Iteration ID: {iteration_result.data['iteration_id']}")
            
            # Perform a search
            search_result = await workflow.search_content("morning gates chapter")
            if search_result.success:
                logger.info(f"Search found {len(search_result.data['results'])} results")
            
        else:
            logger.error(f"Example workflow failed: {result.message}")
            
    except Exception as e:
        logger.error(f"Example workflow error: {e}")


def run_api_server():
    """Run the FastAPI server"""
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
    """Run the CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Book Publication Workflow")
    parser.add_argument("--mode", choices=["api", "cli", "example"], default="api",
                       help="Run mode: api (FastAPI server), cli (command line), example (demo)")
    parser.add_argument("--url", type=str, help="URL to process")
    parser.add_argument("--session", type=str, help="Session name")
    
    args = parser.parse_args()
    
    if args.mode == "api":
        run_api_server()
    elif args.mode == "cli":
        if args.url:
            asyncio.run(run_cli_workflow(args.url, args.session))
        else:
            print("Please provide a URL with --url for CLI mode")
    elif args.mode == "example":
        asyncio.run(run_example_workflow())


async def run_cli_workflow(url: str, session_name: str = None):
    """Run workflow from CLI"""
    try:
        logger.info(f"Processing URL: {url}")
        
        # Initialize the workflow first
        await workflow.initialize()
        
        result = await workflow.process_url(url, session_name)
        
        if result.success:
            print(f"✅ Success! Session ID: {result.data['session_id']}")
            print(f"📊 Quality Score: {result.data['quality_score']:.3f}")
            print(f"⏱️  Processing Time: {result.data['processing_time']:.2f}s")
            
            # Start iteration
            content_id = result.data['processed_content_id']
            iteration_result = await workflow.start_iteration(content_id)
            
            if iteration_result.success:
                print(f"🔄 Iteration started: {iteration_result.data['iteration_id']}")
                print("💡 Provide feedback using the API or web interface")
        else:
            print(f"❌ Failed: {result.message}")
            
    except Exception as e:
        logger.error(f"CLI workflow error: {e}")
        print(f"❌ Error: {e}")
    finally:
        # Clean up resources
        await workflow.cleanup()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Default: run API server
        run_api_server() 