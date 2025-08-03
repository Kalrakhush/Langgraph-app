# src/services/groq_service.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional, Union
from ..config import Config

class GroqService:
    """Service for interacting with Groq API using LangChain."""
    
    def __init__(self):
        self.llm = ChatGroq(
            model=Config.MODEL_NAME,
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0.1,
            max_tokens=1000
        )
    
    def invoke(self, messages: Union[List[Dict[str, str]], List]) -> Dict[str, str]:
        """
        Invoke the Groq model with messages.
        
        Args:
            messages: List of message dictionaries or LangChain message objects
            
        Returns:
            Response with content
        """
        try:
            # Convert dict messages to LangChain message objects if needed
            if messages and isinstance(messages[0], dict):
                langchain_messages = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        langchain_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        langchain_messages.append(AIMessage(content=content))
                    else:
                        langchain_messages.append(HumanMessage(content=content))
                
                messages = langchain_messages
            
            # Invoke the model
            response = self.llm.invoke(messages)
            
            return {"content": response.content}
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return {"content": "I apologize, but I encountered an error processing your request."}
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Optional[str]:
        """
        Get chat completion from Groq API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            Response content or None if error
        """
        try:
            # Update temperature if different from default
            if temperature != 0.1:
                self.llm.temperature = temperature
            
            result = self.invoke(messages)
            return result.get("content")
            
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return None