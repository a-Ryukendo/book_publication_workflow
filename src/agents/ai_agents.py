import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
import openai
import google.generativeai as genai
from anthropic import Anthropic
from models import ScrapedContent, ProcessedContent, FeedbackType
from config.settings import settings, LLMProvider
from rl.reward_system import ContentProcessingRewardSystem

class BaseAIAgent:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.llm_config = settings.get_llm_config()
        self.provider = settings.llm_provider
        self.reward_system = ContentProcessingRewardSystem()
        self._init_llm()
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_time": 0.0,
            "avg_quality": 0.0
        }

    def _init_llm(self):
        if self.provider == LLMProvider.OPENAI:
            if not self.llm_config["api_key"]:
                raise ValueError("OpenAI API key not set")
            self.client = openai.OpenAI(api_key=self.llm_config["api_key"])
        elif self.provider == LLMProvider.GEMINI:
            if not self.llm_config["api_key"]:
                raise ValueError("Gemini API key not set")
            genai.configure(api_key=self.llm_config["api_key"])
            self.model = genai.GenerativeModel('gemini-pro')
        elif self.provider == LLMProvider.ANTHROPIC:
            if not self.llm_config["api_key"]:
                raise ValueError("Anthropic API key not set")
            self.client = Anthropic(api_key=self.llm_config["api_key"])

    async def _call_llm(self, prompt: str, model: str = None) -> str:
        start = time.time()
        try:
            logger.info(f"LLM call: {self.agent_type} ({self.provider})")
            if self.provider == LLMProvider.OPENAI:
                model = model or self.llm_config["models"].get(self.agent_type, "gpt-4o")
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.7
                )
                result = response.choices[0].message.content
            elif self.provider == LLMProvider.GEMINI:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=4000,
                        temperature=0.7
                    )
                )
                result = response.text
            elif self.provider == LLMProvider.ANTHROPIC:
                model = model or self.llm_config["models"].get(self.agent_type, "claude-3-sonnet-20240229")
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=model,
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
            self._update_stats(True, time.time() - start)
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            self._update_stats(False, time.time() - start)
            raise

    def _update_stats(self, success: bool, elapsed: float):
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
        total = self.stats["total_requests"]
        avg = self.stats["avg_time"]
        self.stats["avg_time"] = ((avg * (total - 1) + elapsed) / total)

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()

class AIWriter(BaseAIAgent):
    def __init__(self):
        super().__init__("writer")
        self.styles = {
            "conservative": "Minimal changes",
            "moderate": "Enhance readability",
            "aggressive": "Rewrite for engagement",
            "creative": "Add storytelling",
            "formal": "Formal style",
            "casual": "Conversational",
            "detailed": "Expand details",
            "concise": "Condense"
        }

    async def write_content(self, scraped: ScrapedContent, style: str = "moderate", custom_prompt: str = None) -> str:
        state = self._state_vector(scraped, style=style)
        rl_style = self.reward_system.get_processing_action(state)
        if style == "auto":
            style = rl_style
        style_instruction = self.styles.get(style, self.styles["moderate"])
        prompt = f"""
        Rewrite the following content.
        Title: {scraped.title}
        Content: {scraped.content}
        Style: {style}
        Instruction: {style_instruction}
        {f'Custom: {custom_prompt}' if custom_prompt else ''}
        Only return the rewritten content.
        """
        try:
            out = await self._call_llm(prompt)
            if not out:
                logger.error("Writer: empty LLM response")
                raise ValueError("LLM returned empty response")
            quality = self._quality(scraped.content, out)
            try:
                reward = self.reward_system.calculate_processing_reward(
                    scraped.content, out, quality, 1
                )
                next_state = self._state_vector(scraped, out, quality, style)
                self.reward_system.update_processing_state(state, style, reward, next_state)
            except Exception as rl_error:
                logger.error(f"RL failed: {rl_error}")
            logger.info(f"Writer done (quality: {quality:.3f})")
            return out
        except Exception as e:
            logger.error(f"Writer failed: {e}")
            raise

    def _state_vector(self, scraped: ScrapedContent, transformed: str = None, quality: float = 0.0, style: str = "moderate") -> List[float]:
        orig_len = len(scraped.content) if scraped.content else 0
        trans_len = len(transformed) if transformed else orig_len
        word_count = len(scraped.content.split()) if scraped.content else 0
        title_len = len(scraped.title) if scraped.title else 0
        has_title = float(bool(scraped.title))
        has_content = float(orig_len > 100)
        style_idx = list(self.styles.keys()).index(style) if style in self.styles else 0
        # 10 elements, all meaningful
        return [
            orig_len / 1000,
            trans_len / 1000,
            quality,
            word_count / 100,
            title_len / 100,
            has_title,
            has_content,
            float(quality > 0.5),
            float(len(scraped.metadata) > 0),
            style_idx / 10.0
        ]

    def _quality(self, original: str, transformed: str) -> float:
        return 1.0 if transformed and original else 0.5

class AIReviewer(BaseAIAgent):
    def __init__(self):
        super().__init__("reviewer")

    async def review_content(self, content: str, title: str = None, focus_areas: List[str] = None) -> Dict[str, Any]:
        return {"recommendations": [], "score": 1.0}

    def _state_vector(self, content: str, score: float = 0.0) -> List[float]:
        length = len(content)
        word_count = len(content.split())
        sentence_count = content.count('.')
        para_count = content.count('\n\n')
        # 10 elements
        return [
            length / 1000,
            word_count / 100,
            sentence_count / 10,
            para_count / 5,
            float(length > 100),
            float(score > 0.5),
            score,
            float(word_count > 50),
            float(sentence_count > 5),
            float(para_count > 1)
        ]

class AIEditor(BaseAIAgent):
    def __init__(self):
        super().__init__("editor")

    async def edit_content(self, content: str, review_feedback: Dict[str, Any] = None, focus_areas: List[str] = None) -> str:
        return content

    def _state_vector(self, original: str, edited: str, score: float = 0.0) -> List[float]:
        orig_len = len(original)
        edit_len = len(edited)
        word_diff = abs(len(original.split()) - len(edited.split()))
        # 10 elements
        return [
            orig_len / 1000,
            edit_len / 1000,
            word_diff / 100,
            float(orig_len > 100),
            float(edit_len > 100),
            float(score > 0.5),
            score,
            float(edit_len > orig_len),
            float(edit_len < orig_len),
            float(word_diff > 10)
        ]

class AgentOrchestrator:
    def __init__(self):
        self.writer = AIWriter()
        self.reviewer = AIReviewer()
        self.editor = AIEditor()

    async def process_content(self, scraped: ScrapedContent, config: Dict[str, Any] = None) -> ProcessedContent:
        writer_out = await self.writer.write_content(scraped, config.get("writing_style", "moderate"))
        reviewer_out = await self.reviewer.review_content(writer_out)
        editor_out = await self.editor.edit_content(writer_out, review_feedback=reviewer_out)
        return ProcessedContent(
            original_content_id=scraped.id,
            writer_output=writer_out,
            reviewer_output=str(reviewer_out),
            editor_output=editor_out,
            quality_score=1.0,
            processing_metadata={},
        )

    def get_agent_stats(self) -> Dict[str, Any]:
        return {
            "writer": self.writer.get_stats(),
            "reviewer": self.reviewer.get_stats(),
            "editor": self.editor.get_stats()
        }

    def save_models(self):
        pass 