import argparse
import logging
import wandb
import json
from pathlib import Path
from typing import List, Dict, Any
import time

from src.data.loader import get_loader
from src.data.preprocessor import get_preprocessor
from src.models.openai_client import get_openai_model
from src.evaluation.answer_extractor import get_answer_extractor
from src.evaluation.metrics import get_evaluation_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

class ModelEvaluator:
    """Evaluates model performance on ConvFinQA dataset"""
    
    def __init__(self, model_name: str = "gpt-4o", use_wandb: bool = True):
        self.loader = get_loader()
        self.preprocessor = get_preprocessor()
        self.model = get_openai_model(model_name)
        self.extractor = get_answer_extractor()
        self.metrics = get_evaluation_metrics(use_wandb=use_wandb)
        
        logger.info(f"Initialised model evaluator with {model_name}")
        
    def evaluate(
        self,
        dataset_file: str,
        max_examples: int = None,
        temperature: float = 0.0,
        max_tokens: int = 100,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """Run evaluation with WandB logging on ConvFinQA dataset"""
        
        # get examples from dataset
        logger.info(f"Loading examples from {dataset_file}")
        examples = self.loader.load_and_extract(dataset_file)
        
        if max_examples and max_examples < len(examples):
            logger.info(f"Limiting evaluation to {max_examples} examples")
            examples = examples[:max_examples]
        
        # preprocess examples
        logger.info(f"Preprocessing {len(examples)} examples")
        processed_examples = self.preprocessor.preprocess_examples(examples)
        
        # generate predictions
        predictions = []
        ground_truths = []
        all_results = []
        
        logger.info(f"Generating predictions for {len(processed_examples)} examples")
        for i, example in enumerate(processed_examples):
            try:
                logger.info(f"Processing example {i+1}/{len(processed_examples)}: {example['id']}")
                
                # generate prediction
                prediction = self.model.generate(
                    prompt=example['prompt'],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # extract percentages from prediction and ground truth
                processed_result = self.extractor.process_example(
                    prediction=prediction,
                    ground_truth=example['answer']
                )
                
                # add example ID and save result
                processed_result['id'] = example['id']
                processed_result['question'] = example['question']
                all_results.append(processed_result)
                
                # log progress
                status = processed_result['status']
                pred_value = processed_result.get('prediction_value')
                gt_value = processed_result.get('ground_truth_value')
                
                logger.info(f"Example {example['id']} - status: {status}")
                logger.info(f"  Prediction: {prediction} → {pred_value}%")
                logger.info(f"  Ground truth: {example['answer']} → {gt_value}%")
                
                # avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing example {example['id']}: {e}")
        
        # calculate metrics
        logger.info("Calculating evaluation metrics")
        evaluation_metrics = self.metrics.calculate_all_metrics(all_results)
        
        if save_results:
            self._save_results(
                dataset_file=dataset_file,
                temperature=temperature,
                max_tokens=max_tokens,
                processed_examples=processed_examples,
                evaluation_metrics=evaluation_metrics,
                all_results=all_results
            )
        
        return evaluation_metrics
    
    def _save_results(
        self,
        dataset_file: str,
        temperature: float,
        max_tokens: int,
        processed_examples: List[Dict[str, Any]],
        evaluation_metrics: Dict[str, Any],
        all_results: List[Dict[str, Any]]
    ) -> None:
        """Save evaluation results to JSON file"""
        output_dir = get_project_root() / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_dir / f"eval_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': self.model.model_name,
                    'dataset': dataset_file,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'num_examples': len(processed_examples)
                },
                'metrics': evaluation_metrics,
                'results': all_results
            }, f, indent=2)
            
        logger.info(f"Saved detailed results to {output_file}")

# for backward compatibility
# def evaluate_model(
#     dataset_file: str,
#     model_name: str = "gpt-4o",
#     max_examples: int = None,
#     temperature: float = 0.0,
#     max_tokens: int = 100,
#     save_results: bool = True
# ) -> Dict[str, Any]:
#     """wrapper for backward compatibility"""
#     evaluator = ModelEvaluator(model_name=model_name)
#     return evaluator.evaluate(
#         dataset_file=dataset_file,
#         max_examples=max_examples,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         save_results=save_results
#     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate model on convfinqa dataset")
    
    parser.add_argument("--dataset", type=str, default="train.json",
                        help="dataset file in data/raw directory")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="openai model to use")
    parser.add_argument("--max-examples", type=int, default=10,
                        help="maximum number of examples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="model temperature")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="maximum tokens for model response")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save detailed results")
    parser.add_argument("--wandb-project", type=str, default="convfinqa-eval",
                        help="weights & biases project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="weights & biases run name")
    
    args = parser.parse_args()
    
    # initialise WandB
    wandb_name = args.wandb_name or f"{args.model}-{args.dataset.split('.')[0]}"
    wandb.init(project=args.wandb_project, name=wandb_name, config={
        "model": args.model,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_examples": args.max_examples
    })
    
    # run evaluation
    try:
        # create evaluator
        evaluator = ModelEvaluator(model_name=args.model)
        
        # evaluate model
        metrics = evaluator.evaluate(
            dataset_file=args.dataset,
            max_examples=args.max_examples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            save_results=not args.no_save
        )
        
        # print summary
        print("\nevaluation results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
            
    except Exception as e:
        logger.error(f"evaluation failed: {e}")
        
    finally:
        # close wandb
        wandb.finish() 