import logging
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

def create_convfinqa_prompt(context: str, question: str) -> str:
    """create a complete prompt with context, question and instruction for convfinqa dataset
    
    args:
        context (str): the assembled context including pre-text, table, and post-text
        question (str): the cleaned question text
        
    returns:
        str: formatted prompt ready for llm consumption
    """
    prompt = f"""Context:
    {context}

    Question: {question}

    Important: you MUST answer with a single percentage (%) value only. No other answer format is valid.
    """
    return prompt


def assemble_context(example: Dict[str, Any], 
                    clean_text_func, 
                    format_table_func) -> str:
    """combine all context pieces into a coherent format
    
    args:
        example: dict containing pre_text, table, and post_text
        clean_text_func: function to clean text data
        format_table_func: function to format table data
        
    returns:
        str: assembled context with all components
    """
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
