import weave
from typing import Dict, Any, List


@weave.op()
def log_evaluation_to_weave(
    example_id: str,
    question: str,
    prompt: str,
    model_response: str,
    ground_truth: str,
    prediction_value: float,
    ground_truth_value: float,
    metrics: Dict[str, float],
):
    """log a single example evaluation to weave evals"""

    eval_obj = {
        "example_id": example_id,
        "question": question,
        "prompt": prompt,
        "response": model_response,
        "ground_truth": ground_truth,
        "prediction_value": prediction_value,
        "ground_truth_value": ground_truth_value,
        # include individual metrics
        "correct": prediction_value == ground_truth_value,
        "absolute_error": (
            abs(prediction_value - ground_truth_value)
            if prediction_value is not None and ground_truth_value is not None
            else None
        ),
    }

    return eval_obj
