import requests
from typing import Dict, Any, Optional
from ..config import Config

class WeatherService:
    """Service for fetching real-time weather data."""
    
    def __init__(self,api_key: Optional[str] = None):
        self.api_key = Config.OPENWEATHERMAP_API_KEY
        self.base_url = Config.WEATHER_BASE_URL
    
    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        """
        Fetch weather data for a given city.
        
        Args:
            city: Name of the city
            
        Returns:
            Weather data dictionary or None if error
        """
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for API-level errors
            if data.get('cod') not in (200, '200'):
                print(f"Weather API error for city '{city}': {data.get('message')}")
                return None

            # Use safe key access with .get() instead of checking required keys
            return {
                'city': data.get('name', city),  # Fallback to input city name
                'country': data.get('sys', {}).get('country', ''),
                'temperature': data.get('main', {}).get('temp'),
                'feels_like': data.get('main', {}).get('feels_like'),
                'humidity': data.get('main', {}).get('humidity'),
                'description': data.get('weather', [{}])[0].get('description', ''),
                'wind_speed': data.get('wind', {}).get('speed')
            }

        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, TypeError, IndexError) as e:
            print(f"Error parsing weather response: {e}")
            return None
    
    def format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data for LLM processing."""
        if not weather_data:
            return "Weather data not available."  
        
        return (
            f"Weather in {weather_data.get('city', 'Unknown')}, {weather_data.get('country', '')}:\n"
            f"- Temperature: {weather_data.get('temperature', 'N/A')}°C (feels like {weather_data.get('feels_like', 'N/A')}°C)\n"
            f"- Condition: {weather_data.get('description', 'Unknown').title()}\n"
            f"- Humidity: {weather_data.get('humidity', 'N/A')}%\n"
            f"- Wind Speed: {weather_data.get('wind_speed', 'N/A')} m/s"
        )