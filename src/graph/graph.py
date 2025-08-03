# src/graph/graph.py
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langsmith import traceable
from .nodes import GraphNodes, GraphState

class AIProcessingGraph:
    """Main LangGraph implementation for the AI pipeline."""
    
    def __init__(self):
        self.nodes = GraphNodes()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Define the workflow
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("classify_intent", self.nodes.classify_intent)
        workflow.add_node("fetch_weather", self.nodes.fetch_weather)
        workflow.add_node("retrieve_from_pdf", self.nodes.retrieve_from_pdf)
        workflow.add_node("generate_response", self.nodes.generate_response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "weather": "fetch_weather",
                "pdf": "retrieve_from_pdf"
            }
        )
        
        # Add edges to response generation
        workflow.add_edge("fetch_weather", "generate_response")
        workflow.add_edge("retrieve_from_pdf", "generate_response")
        
        # Add final edge
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _route_by_intent(self, state: GraphState) -> str:
        """Route based on classified intent."""
        return state["intent"]
    
    @traceable(name="ai_pipeline_process")
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the AI pipeline.
        
        Args:
            query: User query string
            
        Returns:
            Processing results with response and metadata
        """
        try:
            # Initialize state
            initial_state: GraphState = {
                "query": query,
                "intent": "",
                "weather_data": {},
                "retrieved_docs": [],
                "final_response": "",
                "metadata": {}
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "query": query,
                "intent": result["intent"],
                "response": result["final_response"],
                "weather_data": result["weather_data"],
                "retrieved_docs_count": len(result["retrieved_docs"]),
                "metadata": result["metadata"]
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": query,
                "intent": "unknown",
                "response": "I encountered an error while processing your request. Please try again.",
                "weather_data": {},
                "retrieved_docs_count": 0,
                "metadata": {"error": str(e)}
            }
    
    async def aprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Async version of process_query.
        
        Args:
            query: User query string
            
        Returns:
            Processing results with response and metadata
        """
        try:
            # Initialize state
            initial_state: GraphState = {
                "query": query,
                "intent": "",
                "weather_data": {},
                "retrieved_docs": [],
                "final_response": "",
                "metadata": {}
            }
            
            # Run the graph asynchronously
            result = await self.graph.ainvoke(initial_state)
            
            return {
                "query": query,
                "intent": result["intent"],
                "response": result["final_response"],
                "weather_data": result["weather_data"],
                "retrieved_docs_count": len(result["retrieved_docs"]),
                "metadata": result["metadata"]
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": query,
                "intent": "unknown",
                "response": "I encountered an error while processing your request. Please try again.",
                "weather_data": {},
                "retrieved_docs_count": 0,
                "metadata": {"error": str(e)}
            }