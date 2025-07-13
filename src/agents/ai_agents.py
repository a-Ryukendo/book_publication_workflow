"""
AI Agents for content processing (Writer, Reviewer, Editor)
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from loguru import logger
import openai
import google.generativeai as genai
from anthropic import Anthropic

from models import ScrapedContent, ProcessedContent, FeedbackType
from config.settings import settings, LLMProvider
from rl.reward_system import ContentProcessingRewardSystem


class BaseAIAgent:
    """Base class for AI agents"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.llm_config = settings.get_llm_config()
        self.provider = settings.llm_provider
        self.reward_system = ContentProcessingRewardSystem()
        
        # Initialize LLM client based on provider
        self._initialize_llm_client()
        
        # Performance tracking
        self.processing_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "average_quality_score": 0.0
        }
    
    def _initialize_llm_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == LLMProvider.OPENAI:
            if not self.llm_config["api_key"]:
                raise ValueError("OpenAI API key not configured")
            self.client = openai.OpenAI(api_key=self.llm_config["api_key"])
            
        elif self.provider == LLMProvider.GEMINI:
            if not self.llm_config["api_key"]:
                raise ValueError("Gemini API key not configured")
            genai.configure(api_key=self.llm_config["api_key"])
            self.model = genai.GenerativeModel('gemini-pro')
            
        elif self.provider == LLMProvider.ANTHROPIC:
            if not self.llm_config["api_key"]:
                raise ValueError("Anthropic API key not configured")
            self.client = Anthropic(api_key=self.llm_config["api_key"])
    
    async def _call_llm(self, prompt: str, model: str = None) -> str:
        """Call LLM with prompt and return response"""
        start_time = time.time()
        
        try:
            logger.info(f"Calling LLM for {self.agent_type} with provider: {self.provider}")
            
            if self.provider == LLMProvider.OPENAI:
                model = model or self.llm_config["models"].get(self.agent_type, "gpt-4o")
                logger.info(f"Using OpenAI model: {model}")
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000,
                    temperature=0.7
                )
                result = response.choices[0].message.content
                logger.info(f"OpenAI response received, length: {len(result) if result else 0}")
                
            elif self.provider == LLMProvider.GEMINI:
                logger.info("Using Gemini model")
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=4000,
                        temperature=0.7
                    )
                )
                result = response.text
                logger.info(f"Gemini response received, length: {len(result) if result else 0}")
                
            elif self.provider == LLMProvider.ANTHROPIC:
                model = model or self.llm_config["models"].get(self.agent_type, "claude-3-sonnet-20240229")
                logger.info(f"Using Anthropic model: {model}")
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=model,
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
                logger.info(f"Anthropic response received, length: {len(result) if result else 0}")
            
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            logger.info(f"LLM call completed for {self.agent_type}, returning result")
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed for {self.agent_type}: {e}")
            self._update_stats(False, time.time() - start_time)
            raise
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["total_requests"] += 1
        if success:
            self.processing_stats["successful_requests"] += 1
        
        # Update average processing time
        total_requests = self.processing_stats["total_requests"]
        current_avg = self.processing_stats["average_processing_time"]
        self.processing_stats["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.processing_stats.copy()


class AIWriter(BaseAIAgent):
    """AI Writer agent for content transformation"""
    
    def __init__(self):
        super().__init__("writer")
        self.writing_styles = {
            "conservative": "Maintain the original tone and style with minimal changes",
            "moderate": "Enhance readability while preserving the core message",
            "aggressive": "Significantly rewrite for better engagement and clarity",
            "creative": "Add creative elements and storytelling techniques",
            "formal": "Convert to formal academic or business writing style",
            "casual": "Convert to conversational and casual writing style",
            "detailed": "Expand with additional details and explanations",
            "concise": "Condense and simplify for brevity"
        }
    
    async def write_content(self, scraped_content: ScrapedContent, 
                          style: str = "moderate", custom_prompt: str = None) -> str:
        """Transform scraped content using AI writing"""
        
        # Get RL-optimized writing strategy
        state_vector = self._create_state_vector(scraped_content, style=style)
        rl_style = self.reward_system.get_processing_action(state_vector)
        
        # Use RL style if no specific style provided
        if style == "auto":
            style = rl_style
        
        style_instruction = self.writing_styles.get(style, self.writing_styles["moderate"])
        
        prompt = f"""
        You are an expert content writer tasked with transforming the following scraped content.
        
        Original Content:
        Title: {scraped_content.title}
        Content: {scraped_content.content}
        
        Writing Style: {style}
        Style Instruction: {style_instruction}
        
        {f"Custom Instructions: {custom_prompt}" if custom_prompt else ""}
        
        Please transform this content according to the specified style while:
        1. Maintaining the core message and key information
        2. Improving readability and engagement
        3. Ensuring proper grammar and structure
        4. Adding appropriate transitions and flow
        5. Preserving important details and context
        
        Return only the transformed content without any additional commentary.
        """
        
        try:
            transformed_content = await self._call_llm(prompt)
            
            # Check if LLM call returned None or empty content
            if not transformed_content:
                logger.error("AI Writer received empty response from LLM")
                raise ValueError("LLM returned empty response")
            
            # Calculate quality score and update RL
            quality_score = self._calculate_writing_quality(scraped_content.content, transformed_content)
            
            try:
                reward = self.reward_system.calculate_processing_reward(
                    scraped_content.content, transformed_content, quality_score, 1
                )
                
                next_state = self._create_state_vector(scraped_content, transformed_content, quality_score, style)
                self.reward_system.update_processing_state(state_vector, style, reward, next_state)
            except Exception as rl_error:
                logger.error(f"RL processing failed: {rl_error}")
                # Continue without RL update
                reward = 0.0
            
            logger.info(f"AI Writer completed transformation with quality score: {quality_score:.3f}")
            return transformed_content
            
        except Exception as e:
            logger.error(f"AI Writer failed: {e}")
            raise
    
    def _create_state_vector(self, scraped_content: ScrapedContent, 
                           transformed_content: str = None, quality_score: float = 0.0, style: str = "moderate") -> List[float]:
        """Create state vector for RL"""
        try:
            logger.info(f"Creating state vector for {self.agent_type}")
            logger.info(f"Scraped content: title='{scraped_content.title}', content_length={len(scraped_content.content) if scraped_content.content else 0}")
            
            original_length = len(scraped_content.content) if scraped_content.content else 0
            transformed_length = len(transformed_content) if transformed_content else original_length
            
            # Create state vector step by step with logging
            logger.info("Creating state vector elements...")
            
            elem1 = original_length / 1000
            logger.info(f"Element 1 (original_length/1000): {elem1}")
            
            elem2 = transformed_length / 1000
            logger.info(f"Element 2 (transformed_length/1000): {elem2}")
            
            elem3 = quality_score
            logger.info(f"Element 3 (quality_score): {elem3}")
            
            elem4 = len(scraped_content.title) / 100 if scraped_content.title else 0
            logger.info(f"Element 4 (title_length/100): {elem4}")
            
            elem5 = len(scraped_content.content.split()) / 100 if scraped_content.content else 0
            logger.info(f"Element 5 (word_count/100): {elem5}")
            
            elem6 = float(bool(scraped_content.title) if scraped_content.title else False)
            logger.info(f"Element 6 (has_title): {elem6}")
            
            elem7 = float(original_length > 100)
            logger.info(f"Element 7 (has_substantial_content): {elem7}")
            
            elem8 = float(quality_score > 0.5)
            logger.info(f"Element 8 (good_quality): {elem8}")
            
            elem9 = float(len(scraped_content.metadata) > 0)
            logger.info(f"Element 9 (has_metadata): {elem9}")
            
            elem10 = abs(transformed_length - original_length) / max(original_length, 1)
            logger.info(f"Element 10 (length_change_ratio): {elem10}")
            
            elem11 = float(transformed_content is not None and len(transformed_content) > 0 if transformed_content else False)
            logger.info(f"Element 11 (has_transformed_content): {elem11}")
            
            elem12 = time.time() % 1000
            logger.info(f"Element 12 (time_component): {elem12}")
            
            elem13 = float(style in ["conservative", "moderate"])
            logger.info(f"Element 13 (conservative_style): {elem13}")
            
            elem14 = float(style in ["aggressive", "creative"])
            logger.info(f"Element 14 (creative_style): {elem14}")
            
            elem15 = float(style in ["formal", "casual"])
            logger.info(f"Element 15 (style_type): {elem15}")
            
            state_vector = [elem1, elem2, elem3, elem4, elem5, elem6, elem7, elem8, elem9, elem10, elem11, elem12, elem13, elem14, elem15]
            
            logger.info(f"State vector created successfully with {len(state_vector)} elements")
            return state_vector
            
        except Exception as e:
            logger.error(f"Error creating state vector: {e}")
            raise
    
    def _calculate_writing_quality(self, original: str, transformed: str) -> float:
        """Calculate quality score for writing transformation"""
        if not transformed:
            return 0.0
        
        # Content preservation score
        original_words = set(original.lower().split())
        transformed_words = set(transformed.lower().split())
        
        if not original_words:
            return 0.0
        
        preservation_score = len(original_words.intersection(transformed_words)) / len(original_words)
        
        # Readability improvement score
        original_sentences = original.count('.') + original.count('!') + original.count('?')
        transformed_sentences = transformed.count('.') + transformed.count('!') + transformed.count('?')
        
        readability_score = min(transformed_sentences / max(original_sentences, 1), 2.0) / 2.0
        
        # Length appropriateness score
        length_ratio = len(transformed) / max(len(original), 1)
        length_score = 1.0 - min(abs(length_ratio - 1.0), 0.5)  # Prefer similar length
        
        # Composite score
        quality_score = (preservation_score * 0.4 + readability_score * 0.3 + length_score * 0.3)
        return min(quality_score, 1.0)


class AIReviewer(BaseAIAgent):
    """AI Reviewer agent for content quality assessment"""
    
    def __init__(self):
        super().__init__("reviewer")
        self.review_criteria = {
            "clarity": "Is the content clear and easy to understand?",
            "accuracy": "Is the information accurate and well-researched?",
            "engagement": "Is the content engaging and interesting to read?",
            "structure": "Is the content well-structured and organized?",
            "grammar": "Is the grammar and spelling correct?",
            "originality": "Is the content original and not plagiarized?",
            "relevance": "Is the content relevant to the intended audience?",
            "completeness": "Is the content complete and comprehensive?"
        }
    
    async def review_content(self, content: str, title: str = None, 
                           focus_areas: List[str] = None) -> Dict[str, Any]:
        """Review content and provide feedback"""
        
        if focus_areas is None:
            focus_areas = list(self.review_criteria.keys())
        
        prompt = f"""
        You are an expert content reviewer. Please review the following content:
        
        Title: {title or "No title provided"}
        Content: {content}
        
        Please evaluate the content based on these criteria:
        {chr(10).join([f"- {area}: {self.review_criteria[area]}" for area in focus_areas])}
        
        For each criterion, provide:
        1. A score from 1-10
        2. Specific feedback and suggestions for improvement
        3. Examples of what works well and what could be improved
        
        Also provide:
        - Overall quality score (1-10)
        - Summary of strengths and weaknesses
        - Specific recommendations for improvement
        - Priority areas that need attention
        
        Return your response as a JSON object with the following structure:
        {{
            "overall_score": <overall_score>,
            "criteria_scores": {{
                "<criterion>": {{
                    "score": <score>,
                    "feedback": "<feedback>"
                }}
            }},
            "strengths": ["<strength1>", "<strength2>"],
            "weaknesses": ["<weakness1>", "<weakness2>"],
            "recommendations": ["<recommendation1>", "<recommendation2>"],
            "priority_areas": ["<priority1>", "<priority2>"]
        }}
        """
        
        try:
            review_response = await self._call_llm(prompt)
            
            # Check if LLM call returned None or empty content
            if not review_response:
                logger.error("AI Reviewer received empty response from LLM")
                raise ValueError("LLM returned empty response")
            
            # Parse JSON response
            try:
                review_data = json.loads(review_response)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', review_response, re.DOTALL)
                if json_match:
                    review_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse review response as JSON")
            
            # Calculate quality metrics
            raw_score = review_data.get("overall_score", 5)
            logger.info(f"Raw overall score from LLM: {raw_score} (type: {type(raw_score)})")
            
            # Ensure score is a number and normalize to 0-1
            if isinstance(raw_score, str):
                try:
                    raw_score = float(raw_score)
                except ValueError:
                    raw_score = 5.0
            
            overall_score = min(max(float(raw_score) / 10.0, 0.0), 1.0)  # Normalize to 0-1 and clamp
            logger.info(f"Normalized overall score: {overall_score}")
            
            # Update the review data with the normalized score
            review_data["overall_score"] = overall_score
            
            criteria_scores = review_data.get("criteria_scores", {})
            
            # Update RL state
            state_vector = self._create_state_vector(content, overall_score, criteria_scores)
            reward = overall_score  # Use overall score as reward
            self.reward_system.update_processing_state(state_vector, "review", reward)
            
            logger.info(f"AI Reviewer completed review with overall score: {overall_score:.3f}")
            return review_data
            
        except Exception as e:
            logger.error(f"AI Reviewer failed: {e}")
            raise
    
    def _create_state_vector(self, content: str, overall_score: float, 
                           criteria_scores: Dict[str, Any]) -> List[float]:
        """Create state vector for RL"""
        return [
            len(content) / 1000,  # Content length (normalized)
            overall_score,  # Overall quality score
            len(content.split()) / 100,  # Word count (normalized)
            content.count('.') / 10,  # Sentence count (normalized)
            content.count('\n\n') / 5,  # Paragraph count (normalized)
            float(len(content) > 100),  # Has substantial content
            float(overall_score > 0.5),  # Good quality
            float(len(criteria_scores) > 0),  # Has detailed criteria
            min(float(criteria_scores.get("clarity", {}).get("score", 5)) / 10, 1.0),  # Clarity score
            min(float(criteria_scores.get("grammar", {}).get("score", 5)) / 10, 1.0),  # Grammar score
            min(float(criteria_scores.get("engagement", {}).get("score", 5)) / 10, 1.0),  # Engagement score
            min(float(criteria_scores.get("structure", {}).get("score", 5)) / 10, 1.0),  # Structure score
            float(overall_score > 0.7),  # High quality
            float(overall_score < 0.3),  # Low quality
            time.time() % 1000  # Time component
        ]


class AIEditor(BaseAIAgent):
    """AI Editor agent for final content refinement"""
    
    def __init__(self):
        super().__init__("editor")
        self.editing_focus_areas = {
            "grammar": "Fix grammar, spelling, and punctuation errors",
            "style": "Improve writing style and tone consistency",
            "clarity": "Enhance clarity and readability",
            "flow": "Improve logical flow and transitions",
            "structure": "Optimize paragraph and section structure",
            "conciseness": "Remove redundancy and improve conciseness",
            "engagement": "Enhance reader engagement and interest",
            "accuracy": "Verify facts and improve accuracy"
        }
    
    async def edit_content(self, content: str, review_feedback: Dict[str, Any] = None,
                          focus_areas: List[str] = None) -> str:
        """Edit content based on review feedback"""
        
        if focus_areas is None:
            focus_areas = list(self.editing_focus_areas.keys())
        
        # Extract priority areas from review feedback
        priority_areas = []
        if review_feedback:
            priority_areas = review_feedback.get("priority_areas", [])
            weaknesses = review_feedback.get("weaknesses", [])
            recommendations = review_feedback.get("recommendations", [])
        
        prompt = f"""
        You are an expert content editor. Please edit the following content:
        
        Content: {content}
        
        Focus Areas: {', '.join(focus_areas)}
        
        {f"Priority Areas from Review: {', '.join(priority_areas)}" if priority_areas else ""}
        {f"Key Weaknesses to Address: {', '.join(weaknesses)}" if review_feedback and 'weaknesses' in review_feedback else ""}
        {f"Specific Recommendations: {', '.join(recommendations)}" if review_feedback and 'recommendations' in review_feedback else ""}
        
        Please edit the content to:
        1. Address the priority areas and weaknesses identified
        2. Implement the specific recommendations provided
        3. Improve overall quality while maintaining the core message
        4. Ensure consistency in style, tone, and formatting
        5. Enhance readability and engagement
        
        For each focus area, apply the following improvements:
        {chr(10).join([f"- {area}: {self.editing_focus_areas[area]}" for area in focus_areas])}
        
        Return the edited content without any additional commentary or explanations.
        """
        
        try:
            edited_content = await self._call_llm(prompt)
            
            # Check if LLM call returned None or empty content
            if not edited_content:
                logger.error("AI Editor received empty response from LLM")
                raise ValueError("LLM returned empty response")
            
            # Calculate improvement score
            improvement_score = self._calculate_improvement_score(content, edited_content, review_feedback)
            
            # Update RL state
            state_vector = self._create_state_vector(content, edited_content, improvement_score)
            reward = improvement_score
            self.reward_system.update_processing_state(state_vector, "edit", reward)
            
            logger.info(f"AI Editor completed editing with improvement score: {improvement_score:.3f}")
            return edited_content
            
        except Exception as e:
            logger.error(f"AI Editor failed: {e}")
            raise
    
    def _calculate_improvement_score(self, original: str, edited: str, 
                                   review_feedback: Dict[str, Any] = None) -> float:
        """Calculate improvement score for editing"""
        if not edited:
            return 0.0
        
        # Content preservation score
        original_words = set(original.lower().split())
        edited_words = set(edited.lower().split())
        
        if not original_words:
            return 0.0
        
        preservation_score = len(original_words.intersection(edited_words)) / len(original_words)
        
        # Readability improvement
        original_sentences = original.count('.') + original.count('!') + original.count('?')
        edited_sentences = edited.count('.') + edited.count('!') + edited.count('?')
        
        readability_improvement = min(edited_sentences / max(original_sentences, 1), 2.0) / 2.0
        
        # Review feedback alignment
        feedback_alignment = 0.5  # Default score
        if review_feedback:
            priority_areas = review_feedback.get("priority_areas", [])
            if priority_areas:
                # Simple heuristic: if content length increased, assume improvements were made
                length_ratio = len(edited) / max(len(original), 1)
                feedback_alignment = min(length_ratio, 1.5) / 1.5
        
        # Composite score
        improvement_score = (
            preservation_score * 0.4 + 
            readability_improvement * 0.3 + 
            feedback_alignment * 0.3
        )
        
        return min(improvement_score, 1.0)
    
    def _create_state_vector(self, original: str, edited: str, improvement_score: float) -> List[float]:
        """Create state vector for RL"""
        return [
            len(original) / 1000,  # Original length (normalized)
            len(edited) / 1000,  # Edited length (normalized)
            improvement_score,  # Improvement score
            len(original.split()) / 100,  # Original word count (normalized)
            len(edited.split()) / 100,  # Edited word count (normalized)
            float(len(original) > 100),  # Has substantial original content
            float(len(edited) > 100),  # Has substantial edited content
            float(improvement_score > 0.5),  # Good improvement
            abs(len(edited) - len(original)) / max(len(original), 1),  # Length change ratio
            float(improvement_score > 0.7),  # High improvement
            float(improvement_score < 0.3),  # Low improvement
            float(len(edited) > len(original)),  # Content expanded
            float(len(edited) < len(original)),  # Content condensed
            float(improvement_score > 0.8),  # Excellent improvement
            time.time() % 1000  # Time component
        ]


class AgentOrchestrator:
    """Orchestrates multiple AI agents for content processing"""
    
    def __init__(self):
        self.writer = AIWriter()
        self.reviewer = AIReviewer()
        self.editor = AIEditor()
        self.reward_system = ContentProcessingRewardSystem()
    
    async def process_content(self, scraped_content: ScrapedContent, 
                            processing_config: Dict[str, Any] = None) -> ProcessedContent:
        """Process content through the full AI pipeline"""
        
        if processing_config is None:
            processing_config = {
                "writing_style": "moderate",
                "review_focus": None,
                "editing_focus": None,
                "custom_prompts": {}
            }
        
        start_time = time.time()
        
        try:
            # Step 1: AI Writer
            logger.info("Starting AI Writer processing...")
            writer_output = await self.writer.write_content(
                scraped_content,
                style=processing_config.get("writing_style", "moderate"),
                custom_prompt=processing_config.get("custom_prompts", {}).get("writer")
            )
            
            # Step 2: AI Reviewer
            logger.info("Starting AI Reviewer processing...")
            review_output = await self.reviewer.review_content(
                writer_output,
                title=scraped_content.title,
                focus_areas=processing_config.get("review_focus")
            )
            
            # Step 3: AI Editor
            logger.info("Starting AI Editor processing...")
            editor_output = await self.editor.edit_content(
                writer_output,
                review_feedback=review_output,
                focus_areas=processing_config.get("editing_focus")
            )
            
            # Calculate overall quality score
            overall_quality = review_output.get("overall_score", 0.5) if review_output else 0.5
            
            # Create processed content object
            processed_content = ProcessedContent(
                original_content_id=scraped_content.id,
                writer_output=writer_output,
                reviewer_output=json.dumps(review_output),
                editor_output=editor_output,
                quality_score=overall_quality,
                processing_metadata={
                    "processing_time": time.time() - start_time,
                    "processing_config": processing_config,
                    "agent_stats": {
                        "writer": self.writer.get_stats(),
                        "reviewer": self.reviewer.get_stats(),
                        "editor": self.editor.get_stats()
                    }
                },
                status="editing"
            )
            
            logger.info(f"Content processing completed with quality score: {overall_quality:.3f}")
            return processed_content
            
        except Exception as e:
            logger.error(f"Content processing failed: {e}")
            raise
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics from all agents"""
        return {
            "writer": self.writer.get_stats(),
            "reviewer": self.reviewer.get_stats(),
            "editor": self.editor.get_stats(),
            "reward_system": self.reward_system.get_performance_stats()
        }
    
    def save_models(self):
        """Save all RL models"""
        self.writer.reward_system.save_model()
        self.reviewer.reward_system.save_model()
        self.editor.reward_system.save_model()
        self.reward_system.save_model() 