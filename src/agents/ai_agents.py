import asyncio
import time
from loguru import logger
import openai
import google.generativeai as genai
from anthropic import Anthropic

from models import Scraped, Processed, Feedback
from config.settings import settings

class WriterAgent:
    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.WRITER_MODEL
        self.openai_key = settings.OPENAI_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY
        self.anthropic_key = settings.ANTHROPIC_API_KEY
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.openai_key)
        elif self.provider == "gemini":
            genai.configure(api_key=self.gemini_key)
            self.gen_model = genai.GenerativeModel('gemini-pro')
        elif self.provider == "anthropic":
            self.client = Anthropic(api_key=self.anthropic_key)
    async def run(self, prompt):
        if self.provider == "openai":
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4000,
                temperature=0.7
            )
            return response.choices[0].message.content
        elif self.provider == "gemini":
            response = await asyncio.to_thread(
                self.gen_model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.7
                )
            )
            return response.text
        elif self.provider == "anthropic":
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

class ReviewerAgent(WriterAgent):
    pass

class EditorAgent(WriterAgent):
    pass

class AgentOrchestrator:
    def __init__(self):
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        self.editor = EditorAgent()
    async def process(self, scraped: Scraped):
        writer_out = await self.writer.run(scraped.body)
        reviewer_out = await self.reviewer.run(writer_out)
        editor_out = await self.editor.run(reviewer_out)
        return Processed(
            original_id=scraped.id,
            writer=writer_out,
            reviewer=reviewer_out,
            editor=editor_out,
            score=0.0,
            meta={},
            finished="",
            status="editing"
        )
    def save_models(self):
        pass
