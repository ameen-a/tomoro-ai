import re
import logging
from typing import Optional, Union, Dict, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PercentageAnswerExtractor:
    """Extracts percentage values from model responses for evaluation"""

    def __init__(self):
        logger.info("Initialised percentage answer extractor")

        # regex for different formats of percentages
        self.percentage_patterns = [
            r"(-?\d+\.?\d*)%",  # matches: 1.3%, -2.5%
            r"(-?\d+\.?\d*)\s*%",  # matches: 1.3 %, -2.5  %
            r"(-?\d+\.?\d*)",  # matches: 1.3, -1.3
            r"(-?\d+\.?\d*)\s*percent",  # matches: 1.3 percent, -2.5 percent
            r"(-?\d+\.?\d*)\s*percentage",  # matches: 1.3 percentage
            r"percentage\s*:\s*(-?\d+\.?\d*)",  # matches: percentage: 1.3
        ]

    def extract_percentage(self, text: str) -> Optional[float]:
        """Get the first percentage value from text"""

        if not text:
            logger.warning("Empty text provided")
            return None

        # try each pattern until a match is found
        for pattern in self.percentage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    percentage = float(match.group(1))
                    logger.info(f"Extracted percentage: {percentage}% from text")
                    return percentage
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to convert matched text to float: {e}")

        logger.warning(f"No percentage value found in text: '{text[:50]}...'")
        return None

    def process_example(self, prediction: str, ground_truth: str) -> Dict[str, Any]:
        """Process a single example by extracting percentages from prediction and ground truth"""
        result = {
            "prediction_raw": prediction,
            "ground_truth_raw": ground_truth,
            "prediction_value": None,
            "ground_truth_value": None,
            "is_valid_prediction": False,
            "is_valid_ground_truth": False,
            "status": "failure",
        }

        # get percentage from prediction
        pred_value = self.extract_percentage(prediction)
        if pred_value is not None:
            result["prediction_value"] = pred_value
            result["is_valid_prediction"] = True

        # get percentage from ground truth
        gt_value = self.extract_percentage(ground_truth)
        if gt_value is not None:
            result["ground_truth_value"] = gt_value
            result["is_valid_ground_truth"] = True

        # set status based on extraction results
        if result["is_valid_prediction"] and result["is_valid_ground_truth"]:
            result["status"] = "success"
        elif not result["is_valid_prediction"]:
            result["status"] = "invalid_prediction"
        else:
            result["status"] = "invalid_ground_truth"

        return result


def get_answer_extractor() -> PercentageAnswerExtractor:
    return PercentageAnswerExtractor()