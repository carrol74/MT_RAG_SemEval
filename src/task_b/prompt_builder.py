"""
build prompts according to MTRAG format, including passages and conversation history
"""

from typing import List, Dict

from src.task_b.config import PromptConfig

class PromptBuilder:
    """Build prompts following MTRAG format"""
    
    def __init__(self, config: PromptConfig):
        self.config = config
    
    def build_prompt(
        self, 
        passages: List[Dict],
        conversation: List[Dict]
    ) -> str:
        """
        Build complete prompt for generation
        
        Args:
            passages: List of passage dictionaries
            conversation: List of conversation turns
            
        Returns:
            Complete prompt string
        """
        # 1. Format passages
        passages_text = self._format_passages(passages)
        
        # 2. Format conversation
        conversation_text = self._format_conversation(conversation)
        
        # 3. Combine using template
        prompt = self.config.template.format(
            max_words=self.config.max_words,
            idk_phrase=self.config.idk_phrase,
            passages=passages_text,
            conversation=conversation_text
        )
        
        return prompt
    
    def _format_passages(self, passages: List[Dict]) -> str:
        """
        Format passages as:
        PASSAGE 1
        <text>
        
        PASSAGE 2
        <text>
        """
        if not passages:
            return "No documents provided."
        
        formatted = []
        for i, passage in enumerate(passages, 1):
            text = passage.get('text', '')
            formatted.append(f"PASSAGE {i}\n{text}")
        
        return "\n\n".join(formatted)
    
    def _format_conversation(self, conversation: List[Dict]) -> str:
        """
        Format conversation as:
        User: <question 1>
        Agent: <answer 1>
        User: <question 2>
        Agent: <answer 2>
        User: <current question>

        Note: No Agent: after the last user turn, to let the model generate response.
        """
        if not conversation:
            return ""
        
        formatted = []
        
        for turn in conversation:
            speaker = turn['speaker'].capitalize()
            text = turn['text']
            formatted.append(f"{speaker}: {text}")
        
        return "\n".join(formatted)
    
    def build_prompt_simple(
        self,
        question: str,
        passages: List[str],
        history: List[Dict] = None
    ) -> str:
        """
        Simplified interface
        
        Args:
            question: Current question text
            passages: List of passage text strings
            history: Optional conversation history
            
        Returns:
            Complete prompt
        """
        # Convert to standard format
        passages_dicts = [{'text': p} for p in passages]
        
        conversation = []
        if history:
            conversation.extend(history)
        conversation.append({'speaker': 'user', 'text': question})
        
        return self.build_prompt(passages_dicts, conversation)


# # Example usage
# if __name__ == "__main__":
#     from src.task_b.config import DEFAULT_CONFIG
    
#     builder = PromptBuilder(DEFAULT_CONFIG.prompt)
    
#     # Example conversation
#     conversation = [
#         {'speaker': 'user', 'text': 'where does doctor strange get his powers from'},
#         {'speaker': 'agent', 'text': 'Doctor Strange gets powers from mystical entities...'},
#         {'speaker': 'user', 'text': 'how many films does he appear in'}
#     ]
    
#     # Example passages
#     passages = [
#         {'text': 'Doctor Strange appears in several Marvel films including...'},
#         {'text': 'The character was first portrayed by Benedict Cumberbatch...'}
#     ]
    
#     prompt = builder.build_prompt(passages, conversation)
#     print(prompt)
#     print("\n" + "="*80 + "\n")