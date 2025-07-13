#!/usr/bin/env python3
"""
Example usage of the Automated Book Publication Workflow
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workflow import workflow
from models import ScrapingRequest, ProcessingRequest, IterationRequest
from loguru import logger


async def example_basic_workflow():
    """Example of basic workflow usage"""
    print("🚀 Example: Basic Workflow")
    print("=" * 50)
    
    # Initialize workflow
    await workflow.initialize()
    
    try:
        # Example URL from requirements
        url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
        
        print(f"📖 Processing URL: {url}")
        
        # Process the URL
        result = await workflow.process_url(
            url=url,
            session_name="example_session",
            include_screenshot=True
        )
        
        if result.success:
            print("✅ Content processed successfully!")
            print(f"📊 Quality Score: {result.data['quality_score']:.3f}")
            print(f"⏱️  Processing Time: {result.data['processing_time']:.2f}s")
            print(f"🆔 Session ID: {result.data['session_id']}")
            
            return result.data['processed_content_id']
        else:
            print(f"❌ Processing failed: {result.message}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def example_human_iteration(content_id: str):
    """Example of human-in-the-loop iteration"""
    print("\n🔄 Example: Human-in-the-Loop Iteration")
    print("=" * 50)
    
    try:
        # Start iteration
        iteration_result = await workflow.start_iteration(content_id, "example_user")
        
        if iteration_result.success:
            print("✅ Iteration started successfully!")
            iteration_id = iteration_result.data['iteration_id']
            print(f"🆔 Iteration ID: {iteration_id}")
            
            # Submit feedback
            feedback_result = await workflow.submit_human_feedback(
                iteration_id=iteration_id,
                feedback="The content needs more engaging opening paragraphs and better flow between sections.",
                feedback_type="reviewer",
                user_id="example_user",
                rating=3.5
            )
            
            if feedback_result.success:
                print("✅ Feedback submitted successfully!")
                print(f"📝 Current content length: {len(feedback_result.data['current_content'])} characters")
                print(f"📊 Iteration status: {feedback_result.data['iteration_status']}")
            else:
                print(f"❌ Feedback submission failed: {feedback_result.message}")
        else:
            print(f"❌ Iteration start failed: {iteration_result.message}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_semantic_search():
    """Example of semantic search"""
    print("\n🔍 Example: Semantic Search")
    print("=" * 50)
    
    try:
        # Search for content
        search_result = await workflow.search_content(
            query="morning gates chapter content",
            filters={"type": "processed"},
            limit=5
        )
        
        if search_result.success:
            print(f"✅ Search completed: {len(search_result.data['results'])} results found")
            
            for i, result in enumerate(search_result.data['results'][:3], 1):
                print(f"\n📄 Result {i}:")
                print(f"   ID: {result['id']}")
                print(f"   Similarity: {result['similarity_score']:.3f}")
                print(f"   Content preview: {result['content'][:100]}...")
                
                if 'metadata' in result:
                    metadata = result['metadata']
                    print(f"   Quality Score: {metadata.get('quality_score', 'N/A')}")
                    print(f"   Type: {metadata.get('type', 'N/A')}")
        else:
            print(f"❌ Search failed: {search_result.message}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_voice_commands():
    """Example of voice command processing"""
    print("\n🎤 Example: Voice Commands")
    print("=" * 50)
    
    try:
        # Record audio (simulated)
        print("🎙️  Recording audio...")
        audio_file = await workflow.voice_processor.record_audio(duration=3)
        print(f"✅ Audio recorded: {audio_file}")
        
        # Process voice command
        voice_result = await workflow.process_voice_command(audio_file)
        
        if voice_result.success:
            print("✅ Voice command processed successfully!")
            print(f"📝 Command: {voice_result.data.get('action', 'Unknown')}")
            print(f"💬 Message: {voice_result.message}")
        else:
            print(f"❌ Voice processing failed: {voice_result.message}")
            
        # Text-to-speech
        print("\n🔊 Converting text to speech...")
        await workflow.voice_processor.speak_text(
            "Voice command processing completed successfully!"
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_version_control(content_id: str):
    """Example of version control"""
    print("\n📚 Example: Version Control")
    print("=" * 50)
    
    try:
        # Create a version
        version_result = await workflow.create_version(
            content_id=content_id,
            change_description="Initial AI processing with moderate style",
            user_id="example_user"
        )
        
        if version_result.success:
            print("✅ Version created successfully!")
            print(f"🆔 Version ID: {version_result.data['version_id']}")
            print(f"📝 Change description: {version_result.data['change_description']}")
            
            # Get version history
            from database.chroma_manager import chroma_manager
            history = chroma_manager.get_version_history(content_id)
            print(f"📚 Total versions: {len(history)}")
            
        else:
            print(f"❌ Version creation failed: {version_result.message}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_statistics():
    """Example of getting system statistics"""
    print("\n📊 Example: System Statistics")
    print("=" * 50)
    
    try:
        # Get workflow stats
        stats = workflow.get_workflow_stats()
        
        print("📈 Workflow Statistics:")
        print(f"   Total Sessions: {stats['total_sessions']}")
        print(f"   Total Content Items: {stats['total_content_items']}")
        print(f"   Average Processing Time: {stats['average_processing_time']:.2f}s")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        
        # Agent stats
        agent_stats = stats.get('agent_stats', {})
        print("\n🤖 Agent Statistics:")
        for agent, agent_data in agent_stats.items():
            print(f"   {agent.title()}:")
            print(f"     Total Requests: {agent_data.get('total_requests', 0)}")
            print(f"     Success Rate: {agent_data.get('successful_requests', 0)/max(agent_data.get('total_requests', 1), 1):.1%}")
            print(f"     Avg Processing Time: {agent_data.get('average_processing_time', 0):.2f}s")
        
        # Voice stats
        voice_stats = stats.get('voice_stats', {})
        print("\n🎤 Voice Statistics:")
        print(f"   Total Voice Inputs: {voice_stats.get('total_voice_inputs', 0)}")
        print(f"   Successful Transcriptions: {voice_stats.get('successful_transcriptions', 0)}")
        print(f"   Average Confidence: {voice_stats.get('average_confidence', 0):.3f}")
        
        # ChromaDB stats
        chroma_stats = stats.get('chroma_stats', {})
        print("\n🗄️  ChromaDB Statistics:")
        print(f"   Total Documents: {chroma_stats.get('total_documents', 0)}")
        print(f"   Average Quality Score: {chroma_stats.get('average_quality_score', 0):.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def example_batch_processing():
    """Example of batch processing multiple URLs"""
    print("\n📦 Example: Batch Processing")
    print("=" * 50)
    
    # Example URLs
    urls = [
        "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1",
        "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_2",
        "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_3"
    ]
    
    try:
        print(f"📖 Processing {len(urls)} URLs in batch...")
        
        results = []
        for i, url in enumerate(urls, 1):
            print(f"   Processing URL {i}/{len(urls)}: {url}")
            
            result = await workflow.process_url(
                url=url,
                session_name=f"batch_session_{i}",
                include_screenshot=True
            )
            
            if result.success:
                print(f"   ✅ Success - Quality: {result.data['quality_score']:.3f}")
                results.append(result.data)
            else:
                print(f"   ❌ Failed - {result.message}")
        
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total URLs: {len(urls)}")
        print(f"   Successful: {len(results)}")
        print(f"   Failed: {len(urls) - len(results)}")
        
        if results:
            avg_quality = sum(r['quality_score'] for r in results) / len(results)
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            print(f"   Average Quality Score: {avg_quality:.3f}")
            print(f"   Average Processing Time: {avg_time:.2f}s")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Main example function"""
    print("🚀 Automated Book Publication Workflow - Example Usage")
    print("=" * 60)
    
    try:
        # Initialize workflow
        await workflow.initialize()
        
        # Run examples
        content_id = await example_basic_workflow()
        
        if content_id:
            await example_human_iteration(content_id)
            await example_version_control(content_id)
        
        await example_semantic_search()
        await example_voice_commands()
        await example_statistics()
        await example_batch_processing()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n💡 Tips:")
        print("- Use the API endpoints for programmatic access")
        print("- Check the API documentation at http://localhost:8000/docs")
        print("- Monitor logs for detailed information")
        print("- Use voice commands for hands-free operation")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        logger.error(f"Example execution error: {e}")
    finally:
        await workflow.cleanup()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 