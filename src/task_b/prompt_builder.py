"""
build prompts according to MTRAG format, including passages and conversation history
"""

from typing import List, Dict

from src.task_b.config import PromptConfig

class PromptBuilder:
    """Build prompts following MTRAG format"""
    
    def __init__(self, config: PromptConfig):
        self.config = config

    def build_prompt_with_system_role(
        self, 
        passages: List[Dict],
        conversation: List[Dict]
    ) -> Dict:
        """
        Build prompt that separates system and user parts
        
        Args:
            passages: List of passage dictionaries
            conversation: List of conversation turns
            
        Returns:
            {
                'system': system_prompt,
                'user': user_prompt
            }
        """
#         # System prompt (instruction only)
#         system_prompt = f"""You are a helpful assistant specialized in answering questions based on provided documents.

# CRITICAL RULES:
# 1. Answer using ONLY information from the PASSAGE sections
# 2. Keep responses under {self.config.max_words} words
# 3. If passages don't contain the answer, say "{self.config.idk_phrase}"
# 4. Be specific and factual
# 5. For multi-turn conversations, understand references like "he", "it" from context
# 6. Do NOT add explanations, follow-up questions, suggestions, or any additional text
# 7. All responses must be in English only."""

#         # User prompt (passages + conversation only)
#         # 1. Format passages
#         passages_text = self._format_passages(passages)
        
#         # 2. Format conversation
#         conversation_text = self._format_conversation(conversation)

#         user_prompt = f"""{passages_text}

# {conversation_text}"""

#         return {
#             'system': system_prompt,
#             'user': user_prompt
#         }

        passages_text = self._format_passages(passages)
        conversation_text = self._format_conversation(conversation)    

        full_prompt = self.config.template.format(
            max_words=self.config.max_words,
            idk_phrase=self.config.idk_phrase,
            passages=passages_text,
            conversation=conversation_text
        )

        if self.config.use_few_shot:
            split_marker = "EXAMPLE 1"
            if split_marker in full_prompt:
                parts = full_prompt.split(split_marker, 1)
                system_prompt = parts[0].strip()
                user_prompt = split_marker + parts[1]
            else:
                system_prompt = ""
                user_prompt = full_prompt
        else:
            lines = full_prompt.split('\n\n', 1)
            if len(lines) > 1:
                system_prompt = lines[0]
                user_prompt = lines[1]
            else:
                system_prompt = ""
                user_prompt = full_prompt
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }

    def build_prompt(
        self, 
        passages: List[Dict],
        conversation: List[Dict]
    ) -> Dict:
        """
        Build complete prompt for generation
        
        Args:
            passages: List of passage dictionaries
            conversation: List of conversation turns
            
        Returns:
            Complete dict format for consistency with build_prompt_with_system_role
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
        
        # Return as dict with empty system (to match build_prompt_with_system_role format)
        return {
            'system': '',
            'user': prompt
        }
    
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
    
#     prompt = builder.build_prompt_with_system_role(passages, conversation)
#     print(prompt)
#     print("\n" + "="*80 + "\n")