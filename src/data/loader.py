import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


class ConvFinQALoader:
    """Extracts question-answer pairs with context from the ConvFinQA dataset"""

    def __init__(self, data_dir: Union[str, Path] = None):
        if data_dir is None:
            # default to project_root/data/raw
            self.data_dir = get_project_root() / "data" / "raw"
        else:
            self.data_dir = Path(data_dir)

        logger.info(f"initialised ConvFinQALoader with directory: {self.data_dir}")

    def load_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load and parse a JSON file from the data directory"""
        file_path = self.data_dir / filename

        if not file_path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading data from {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Successfully loaded {len(data)} examples from {filename}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {file_path}: {e}")
            raise

    def extract_qa_with_context(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract question-answer pairs with context from the loaded data"""
        qa_with_context = []

        logger.info("Extracting question-answer pairs with context")
        for item in data:
            try:
                # only process items with QA entries
                if "qa" in item:
                    # extract all required fields
                    example = {
                        "id": item["id"],
                        "question": item["qa"]["question"],
                        "answer": item["qa"]["answer"],
                        "pre_text": item.get("pre_text", []),
                        "post_text": item.get("post_text", []),
                        "table": item.get("table", []),
                    }

                    qa_with_context.append(example)

            except KeyError as e:
                logger.warning(f"Skipping example {item['id']}: missing key {e}")
            except Exception as e:
                logger.warning(f"Error processing example {item['id']}: {e}")

        logger.info(
            f"Extracted {len(qa_with_context)} question-answer pairs with context"
        )
        return qa_with_context

    def load_and_extract(self, filename: str = "train.json") -> List[Dict[str, Any]]:
        """Convenience method to load file and extract QA pairs with context in one step"""
        data = self.load_file(filename)
        return self.extract_qa_with_context(data)


def get_loader() -> ConvFinQALoader:
    return ConvFinQALoader()


if __name__ == "__main__":
    # test the loader
    loader = get_loader()
    try:
        # load train.json and extract QA pairs with context
        examples = loader.load_and_extract("train.json")

        # print a few examples
        print(f"\nfound {len(examples)} question-answer pairs with context")
        print("\nexample QA with context:")
        for i, example in enumerate(examples[:1]):
            print(f"\nexample {i+1}:")
            print(f"id: {example['id']}")
            print(f"question: {example['question']}")
            print(f"answer: {example['answer']}")
            print(
                f"pre-text: {example['pre_text'][:2] if example['pre_text'] else '[]'}"
            )
            print(
                f"post-text: {example['post_text'][:2] if example['post_text'] else '[]'}"
            )
            print(f"table: {len(example['table'])} rows")

    except Exception as e:
        logger.error(f"error in test run: {e}")
