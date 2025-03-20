import logging
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

def create_convfinqa_prompt(context: str, question: str) -> str:
    """Create a complete prompt with context, question and instruction for ConvFinQA dataset"""
    prompt = f"""Context:
    {context}

    Question: {question}

    Important: you MUST answer with a single percentage (%) value only. No other answer format is valid.
    """
    return prompt


def assemble_context(example: Dict[str, Any], 
                    clean_text_func, 
                    format_table_func) -> str:
    """Combine all context pieces into a coherent format"""
    context_parts = []
    
    # pre-text 
    if example.get('pre_text'):
        pre_text = clean_text_func(example['pre_text'])
        if pre_text:
            context_parts.append("Pre-text:\n" + pre_text)
    
    # table
    if example.get('table'):
        try:
            table_str = format_table_func(example['table'])
            if table_str:
                context_parts.append("Table:\n" + table_str)
        except Exception as e:
            logger.warning(f"error formatting table: {e}")
    
    # post-text
    if example.get('post_text'):
        post_text = clean_text_func(example['post_text'])
        if post_text:
            context_parts.append("Post-text:\n" + post_text)
            
    # join pre-text, table and post-text
    return "\n\n".join(context_parts)
