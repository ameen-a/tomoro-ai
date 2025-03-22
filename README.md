# tomoro-ai: ConvFinQA Evaluation

 A tool for evaluating LLMs on the `ConvFinQA` dataset - a conversational financial QA benchmark. 

 ## Setup
### Prerequisites
- Python 3.10+
- OpenAI and/or Anthropic API keys
- W&B account (optional)

### Installation
```bash
git clone https://github.com/your-username/convfinqa-eval.git
cd convfinqa-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Be sure to create a `.env` file in the root directory and add your LLM API keys (OpenAI and/or Anthropic) as well as your W&B API key.


### Usage

```bash
python -m src.evaluation.evaluate --dataset train.json --model gpt-4o --max-examples 10
```

### Parameters
- `dataset`: Dataset file in `data/raw` directory (default: `train.json`)
- `model`: Model to evaluate (default: `claude-3-7-sonnet-20250219`)
- `max-examples`: Maximum number of examples to evaluate (default: `10`)
- `temperature`: Model temperature (default: `0.0`)
- `max-tokens`: Maximum tokens for model response (default: `100`)
- `no-save`: Don't save detailled results
- `wandb-project`: Weights & Biases project name (default: `convfinqa-eval`)
- `wandb-name`: Weights & Biases run name


## Metrics 
- `Exact Match Accuracy`: % of predictions matching ground truth exactly
- `Threshold Accuracy`: % of predictions within threshold (±0.1%, ±0.5%, ±1.0%)
- `MAE`: Average absolute difference
- `MSE`: Average squared difference
- `MRE`: Average relative difference as % of ground truth
