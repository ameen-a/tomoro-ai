import os
import logging
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
import weave

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenAIModel:
    """Wrapper for OpenAI models to generate text responses"""

    def __init__(self, model_name: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OpenAI API key not found in environment variables"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialised OpenAI model with {model_name}")

    @weave.op()
    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 100
    ) -> str:
        """Get LLM response for a given prompt"""
        try:
            logger.info(f"Generating response with model {self.model_name}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            generated_text = response.choices[0].message.content.strip()

            logger.info(f"Successfully generated response")
            return generated_text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def batch_generate(
        self, prompts: List[str], temperature: float = 0.0, max_tokens: int = 100
    ) -> List[str]:
        """Get LLM responses for a batch of prompts"""
        responses = []

        logger.info(f"Batch generating responses for {len(prompts)} prompts")

        for i, prompt in enumerate(prompts):
            try:
                response = self.generate(
                    prompt=prompt, temperature=temperature, max_tokens=max_tokens
                )
                responses.append(response)
                logger.info(f"Completed {i+1}/{len(prompts)} prompts")

            except Exception as e:
                logger.error(f"Error generating response for prompt {i}: {e}")
                responses.append("")  # empty string for failed responses

        return responses


def get_openai_model(model_name: str = "gpt-4o") -> OpenAIModel:
    return OpenAIModel(model_name=model_name)