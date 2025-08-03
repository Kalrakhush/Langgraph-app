# src/graph/nodes.py
from typing import Dict, Any, List, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from ..services.groq_service import GroqService
from ..services.weather_services import WeatherService
from ..services.vector_store import VectorStore

class GraphState(TypedDict):
    """State object for LangGraph using TypedDict."""
    query: str
    intent: str  # "weather" or "pdf"
    weather_data: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    final_response: str
    metadata: Dict[str, Any]

class GraphNodes:
    """Node functions for LangGraph workflow."""
    
    def __init__(self):
        self.llm = GroqService()
        self.weather_service = WeatherService()
        self.vector_store = VectorStore()
    
    def classify_intent(self, state: GraphState) -> Dict[str, Any]:
        """Classify user intent as weather or PDF query."""
        try:
            messages = [
                SystemMessage(content="""You are an intent classifier. Classify the user query as either:
            - "weather": if asking about weather, temperature, climate conditions
            - "pdf": if asking about document content, information from files

            Respond with only one word: "weather" or "pdf" """),
                HumanMessage(content=state["query"])
            ]
            
            response = self.llm.invoke(messages)
            intent = response["content"].strip().lower()
            if intent not in ["weather", "pdf"]:
                intent = "pdf"
            
            return {
                "intent": intent,
                "metadata": {"intent_classification": intent}
            }

        except Exception as e:
            print(f"Error in intent classification: {e}")
            return {
                "intent": "pdf",
                "metadata": {"intent_classification": "error"}
            }

    def fetch_weather(self, state: GraphState) -> Dict[str, Any]:
        """Fetch weather data for the query."""
        try:
            messages = [
                SystemMessage(content="""Extract the city name from this weather query. 
                    Respond with only the city name, nothing else.
                    If no city is mentioned, respond with "London" as default."""),
                HumanMessage(content=state["query"])
            ]
            response = self.llm.invoke(messages)
            city = response["content"].strip()

            weather_data = self.weather_service.get_weather(city)
            return {"weather_data": weather_data or {}}

        except Exception as e:
            print(f"Error fetching weather: {e}")
            return {"weather_data": {}}

    def retrieve_from_pdf(self, state: GraphState) -> Dict[str, Any]:
        """Retrieve relevant documents from PDF collection."""
        try:
            results = self.vector_store.search_similar(state["query"], limit=3)
            return {"retrieved_docs": results}

        except Exception as e:
            print(f"Error retrieving from PDF: {e}")
            return {"retrieved_docs": []}

    def generate_response(self, state: GraphState) -> Dict[str, Any]:
        """Generate final response based on intent and retrieved data."""
        try:
            if state["intent"] == "weather":
                response = self._generate_weather_response(state)
            else:
                response = self._generate_pdf_response(state)

            return {"final_response": response}

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "final_response": "I apologize, but I encountered an error while processing your request."
            }

    def _generate_weather_response(self, state: GraphState) -> str:
        """Generate response for weather queries."""
        if not state["weather_data"]:
            return "I couldn't retrieve weather information. Please check the city name and try again."

        weather_text = self.weather_service.format_weather_response(state["weather_data"])
        messages = [
            SystemMessage(content="""You are a helpful weather assistant. Provide a natural, conversational response 
about the weather based on the provided data. Be concise but informative."""),
            HumanMessage(content=f"User query: {state['query']}\n\nWeather data:\n{weather_text}")
        ]
        response = self.llm.invoke(messages)
        return response["content"]

    def _generate_pdf_response(self, state: GraphState) -> str:
        """Generate response for PDF queries."""
        if not state["retrieved_docs"]:
            return "I couldn't find relevant information in the document. Please try rephrasing your question."

        context = "\n\n".join([doc["text"] for doc in state["retrieved_docs"]])
        messages = [
            SystemMessage(content="""You are a helpful assistant that answers questions based on provided document context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information, say so clearly."""),
            HumanMessage(content=f"Question: {state['query']}\n\nContext:\n{context}")
        ]
        response = self.llm.invoke(messages)
        return response["content"]