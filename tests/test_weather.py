import pytest
import requests
from unittest.mock import Mock, patch
from src.services.weather_services import WeatherService

class TestWeatherService:
    def setup_method(self):
        # Use a dummy API key to avoid hitting the real API
        self.weather_service = WeatherService(api_key='dummy')

    @patch('src.services.weather_services.requests.get')
    def test_get_weather_success(self, mock_get):
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'name': 'London',
            'sys': {'country': 'GB'},
            'main': {
                'temp': 15.5,
                'feels_like': 14.2,
                'humidity': 65
            },
            'weather': [{'description': 'cloudy'}],
            'wind': {'speed': 3.2}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.weather_service.get_weather("London")

        assert result is not None
        assert result['city'] == 'London'
        assert result['country'] == 'GB'
        assert result['temperature'] == 15.5
        assert result['feels_like'] == 14.2
        assert result['humidity'] == 65
        assert result['description'] == 'cloudy'
        assert result['wind_speed'] == 3.2

    @patch('src.services.weather_services.requests.get')
    def test_get_weather_api_error(self, mock_get):
        # Simulate a network/API error
        mock_get.side_effect = requests.RequestException("API Error")

        result = self.weather_service.get_weather("InvalidCity")

        assert result is None

    def test_format_weather_response(self):
        # Provide a complete weather data dict
        weather_data = {
            'city': 'Paris',
            'country': 'FR',
            'temperature': 20.0,
            'feels_like': 19.5,
            'humidity': 70,
            'description': 'sunny',
            'wind_speed': 2.1
        }

        formatted = self.weather_service.format_weather_response(weather_data)

        # Check that key pieces are in the formatted string
        assert 'Paris, FR' in formatted
        assert '20.0°C' in formatted
        assert 'feels like 19.5°C' in formatted
        assert 'Sunny' in formatted  # Capitalized description
        assert 'Humidity: 70%' in formatted
        assert 'Wind Speed: 2.1 m/s' in formatted

    @patch('src.services.weather_services.requests.get')
    def test_missing_fields_in_response(self, mock_get):
        # Mock a response with missing fields
        mock_response = Mock()
        mock_response.json.return_value = {
            'name': 'Unknown',
            # sys missing
            'main': {},
            'weather': [{}],
            'wind': {}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.weather_service.get_weather("CityWithMissingData")

        # Should return a dict (not None) even if fields are missing
        assert isinstance(result, dict)
        assert result['city'] == 'Unknown'
        assert result['country'] is None
        assert result['temperature'] is None
        assert result['feels_like'] is None
        assert result['humidity'] is None
        assert result['description'] is None
        assert result['wind_speed'] is None

        # Formatting should handle None gracefully
        formatted = self.weather_service.format_weather_response(result)
        assert 'Unknown, None' in formatted
        assert 'None°C' in formatted
