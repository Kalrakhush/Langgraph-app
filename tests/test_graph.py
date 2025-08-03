# tests/test_graph.py
import pytest
from unittest.mock import Mock, patch
from src.graph.graph import AIProcessingGraph
from src.graph.nodes import GraphState

class TestAIProcessingGraph:
    
    def setup_method(self):
        self.graph = AIProcessingGraph()
    
    @patch('src.graph.nodes.GraphNodes.classify_intent')
    @patch('src.graph.nodes.GraphNodes.fetch_weather')
    @patch('src.graph.nodes.GraphNodes.generate_response')
    def test_weather_query_flow(self, mock_generate, mock_weather, mock_classify):
        # Mock the flow for weather query
        def mock_classify_fn(state):
            state.intent = "weather"
            return state
        
        def mock_weather_fn(state):
            state.weather_data = {"city": "London", "temperature": 15}
            return state
        
        def mock_generate_fn(state):
            state.final_response = "It's 15°C in London"
            return state
        
        mock_classify.side_effect = mock_classify_fn
        mock_weather.side_effect = mock_weather_fn
        mock_generate.side_effect = mock_generate_fn
        
        result = self.graph.process_query("What's the weather in London?")
        
        assert result["intent"] == "weather"
        assert "London" in result["response"]
    
    def test_invalid_query_handling(self):
        result = self.graph.process_query("")
        
        assert "response" in result
        assert result["intent"] in ["weather", "pdf", "unknown"]