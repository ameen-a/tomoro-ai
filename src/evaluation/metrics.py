import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import wandb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Calculates eval metrics for percentage prediction tasks"""

    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb
        logger.info(f"Initialised evaluation metrics calculator (wandb: {use_wandb})")

    def exact_match_accuracy(
        self, predictions: List[float], ground_truths: List[float]
    ) -> float:
        """Calculate exact match accuracy (if predictions identical to ground truths)"""
        if not predictions or not ground_truths:
            logger.warning("Empty predictions or ground truths")
            return 0.0

        exact_matches = sum(
            1 for p, gt in zip(predictions, ground_truths) if p == gt
        )  # count exact matches
        accuracy = exact_matches / len(predictions)

        logger.info(f"Exact match accuracy: {accuracy:.4f}")
        return accuracy

    def threshold_accuracy(
        self,
        predictions: List[float],
        ground_truths: List[float],
        threshold: float = 0.1,
    ) -> float:
        """Calculate accuracy within a threshold (absolute difference <= threshold)"""
        if not predictions or not ground_truths:
            logger.warning("Empty predictions or ground truths")
            return 0.0

        # example: if threshold is 0.1, then check if the difference
        # between the prediction and ground truth is 10% or less
        within_threshold = sum(
            1 for p, gt in zip(predictions, ground_truths) if abs(p - gt) <= threshold
        )
        accuracy = within_threshold / len(predictions)

        logger.info(f"Threshold accuracy (+/-{threshold}): {accuracy:.4f}")
        return accuracy

    def mean_absolute_error(
        self, predictions: List[float], ground_truths: List[float]
    ) -> float:
        """Calculate MAE between predictions and ground truths"""
        if not predictions or not ground_truths:
            logger.warning("Empty predictions or ground truths")
            return float("inf")

        abs_errors = [abs(p - gt) for p, gt in zip(predictions, ground_truths)]
        mae = sum(abs_errors) / len(predictions)

        logger.info(f"Mean absolute error: {mae:.4f} percentage points")
        return mae

    def mean_squared_error(
        self, predictions: List[float], ground_truths: List[float]
    ) -> float:
        """Calculate MSE between predictions and ground truths"""
        if not predictions or not ground_truths:
            logger.warning("Empty predictions or ground truths")
            return float("inf")

        squared_errors = [(p - gt) ** 2 for p, gt in zip(predictions, ground_truths)]
        mse = sum(squared_errors) / len(predictions)

        logger.info(f"Mean squared error: {mse:.4f} percentage points")
        return mse

    def mean_relative_error(
        self,
        predictions: List[float],
        ground_truths: List[float],
        epsilon: float = 1e-10,
    ) -> float:
        """Calculate mean relative error between predictions and ground truths"""
        if not predictions or not ground_truths:
            logger.warning("Empty predictions or ground truths")
            return float("inf")

        # handle division by zero with epsilon
        relative_errors = [
            abs(p - gt) / (abs(gt) + epsilon)
            for p, gt in zip(predictions, ground_truths)
        ]
        mre = sum(relative_errors) / len(predictions) * 100  # convert to %

        logger.info(f"Mean relative error: {mre:.4f}%")
        return mre

    def calculate_all_metrics(
        self,
        processed_results: List[Dict[str, Any]],
        thresholds: List[float] = [0.1, 0.5, 1.0],
    ) -> Dict[str, float]:
        """Calculate exact match, threshold, MAE, MSE and MRE"""
        # get only valid results
        valid_results = [r for r in processed_results if r.get("status") == "success"]

        if not valid_results:
            logger.warning("No valid results to evaluate")
            return {}

        # get results
        predictions = [r.get("prediction_value") for r in valid_results]
        ground_truths = [r.get("ground_truth_value") for r in valid_results]

        # calculate metrics
        metrics = {
            "num_examples": len(processed_results),
            "num_valid": len(valid_results),
            "valid_percentage": len(valid_results) / len(processed_results) * 100,
            "exact_match_accuracy": self.exact_match_accuracy(
                predictions, ground_truths
            ),
            "mae": self.mean_absolute_error(predictions, ground_truths),
            "mse": self.mean_squared_error(predictions, ground_truths),
            "mre": self.mean_relative_error(predictions, ground_truths),
        }

        # add threshold accuracies
        for threshold in thresholds:
            key = f"threshold_accuracy_{threshold}"
            metrics[key] = self.threshold_accuracy(
                predictions, ground_truths, threshold
            )

        # log metrics to WandB
        if self.use_wandb:
            try:
                wandb.log(metrics)
                logger.info("Successfully logged metrics to WandB")
            except Exception as e:
                logger.error(f"Failed to log metrics to WandB: {e}")

        return metrics


def get_evaluation_metrics(use_wandb: bool = True) -> EvaluationMetrics:
    return EvaluationMetrics(use_wandb=use_wandb)