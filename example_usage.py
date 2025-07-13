import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from workflow import workflow
from loguru import logger

async def basic():
    print("Basic workflow\n" + "="*50)
    await workflow.init()
    url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    print(f"Processing URL: {url}")
    result = await workflow.handle_url(url, session="example", screenshot=True)
    if result.success:
        print("Content processed")
        print(f"Score: {result.data['quality_score']}")
        print(f"Time: {result.data['processing_time']}s")
        print(f"Session: {result.data['session_id']}")
        return result.data['processed_content_id']
    else:
        print(f"Failed: {result.message}")
        return None

async def iterate(cid):
    print("\nHuman feedback\n" + "="*50)
    iteration = await workflow.start_iteration(cid, "user")
    if iteration.success:
        print("Iteration started")
        iid = iteration.data['iteration_id']
        feedback = await workflow.submit_human_feedback(
            iteration_id=iid,
            feedback="Needs more engaging intro.",
            feedback_type="reviewer",
            user_id="user",
            rating=4.0
        )
        if feedback.success:
            print("Feedback submitted")
            print(f"Length: {len(feedback.data['current_content'])}")
            print(f"Status: {feedback.data['iteration_status']}")
        else:
            print(f"Feedback failed: {feedback.message}")
    else:
        print(f"Iteration failed: {iteration.message}")

async def search():
    print("\nSemantic search\n" + "="*50)
    result = await workflow.search_content(query="morning gates chapter", filters={"type": "processed"}, limit=5)
    if result.success:
        print(f"Search: {len(result.data['results'])} found")
        for i, r in enumerate(result.data['results'][:3], 1):
            print(f"{i}. {r}")
    else:
        print(f"Search failed: {result.message}")
