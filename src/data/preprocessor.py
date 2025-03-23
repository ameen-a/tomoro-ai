import logging
from typing import Dict, List, Any, Union

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConvFinQAPreprocessor:
    """Preprocesses data from the ConvFinQA dataset for LLM consumption"""

    def __init__(self):
        logger.info("Initialised ConvFinQA preprocessor")

    def format_table(self, table_data: List[Any]) -> str:
        """Convert JSON table data to LLM-friendly text format"""
        # table is a list of lists
        table_str = ""
        for row in table_data:
            table_str += " | ".join([str(cell) for cell in row]) + "\n"
        return table_str

    def clean_text(self, text: Union[str, List[str]]) -> str:
        """Cleans and normalises text for LLM consumption"""
        if isinstance(text, list):
            # join list of strings into a single string
            text = " ".join([str(item) for item in text])

        # basic processing
        text = str(text)
        cleaned = text.strip()
        cleaned = " ".join(cleaned.split())  # remove extra whitespace

        return cleaned

    def assemble_context(self, example: Dict[str, Any]) -> str:
        """Combine all context pieces into a coherent format"""
        from src.prompts.templates import assemble_context

        return assemble_context(example, self.clean_text, self.format_table)

    def create_prompt(self, example: Dict[str, Any]) -> str:
        """Create a complete prompt with context, question and instruction"""
        context = self.assemble_context(example)
        question = self.clean_text(example["question"])
        from src.prompts.templates import create_convfinqa_prompt

        return create_convfinqa_prompt(context, question)

    def preprocess_examples(
        self, examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Preprocesses a batch of examples for LLM consumption"""
        preprocessed = []

        logger.info(f"Preprocessing {len(examples)} examples")

        for example in examples:
            try:
                processed = {
                    "id": example["id"],
                    "question": self.clean_text(example["question"]),
                    "answer": self.clean_text(example["answer"]),
                    "context": self.assemble_context(example),
                    "prompt": self.create_prompt(example),
                }
                preprocessed.append(processed)
                logger.info(f"Successfully preprocessed example {example['id']}")

            except Exception as e:
                logger.warning(
                    f"Error preprocessing example {example.get('id', 'unknown')}: {e}"
                )

        logger.info(f"Successfully preprocessed {len(preprocessed)} examples")
        return preprocessed


def get_preprocessor() -> ConvFinQAPreprocessor:
    return ConvFinQAPreprocessor()


if __name__ == "__main__":
    # fix import paths for direct script execution
    import sys
    from pathlib import Path

    # add project root to path to allow imports to work
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # now we can run the debug code
    from loader import get_loader

    loader = get_loader()
    preprocessor = get_preprocessor()

    try:
        # load a few examples
        examples = loader.load_and_extract("train.json")[:100]  # try more examples

        # preprocess them
        preprocessed = preprocessor.preprocess_examples(examples)

        # print a preprocessed example - check if we have any
        if preprocessed:
            print("\npreprocessed example:")
            print(f"id: {preprocessed[0]['id']}")
            print(f"question: {preprocessed[0]['question']}")
            print(f"answer: {preprocessed[0]['answer']}")
            print("\nprompt:")
            print(preprocessed[0]["prompt"])
        else:
            print("\nno examples were successfully preprocessed")

            # print details of first example to debug
            print("\nfirst example structure:")
            example = examples[0]
            print(f"id: {example['id']}")
            print(f"question type: {type(example.get('question'))}")
            print(f"table type: {type(example.get('table'))}")
            if example.get("table"):
                print(f"table item type: {type(example['table'][0])}")

    except Exception as e:
        logger.error(f"Error in test run: {e}")
