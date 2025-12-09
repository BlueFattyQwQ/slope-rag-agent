import requests
from typing import Dict, Any
from app.core.config import settings

class WeatherTool:
    def query(self, city: str, date: str = None) -> Dict[str, Any]:
        """
        查询天气信息。
        如果 API Key 是 mock_key，则返回模拟数据。
        """
        if settings.WEATHER_API_KEY == "mock_key":
            return {
                "city": city,
                "date": date or "today",
                "condition": "Heavy Rain",
                "temperature_c": 22.5,
                "precip_mm": 55.0,
                "is_mock": True
            }
        
        # 真实 API 调用 (示例)
        try:
            url = f"{settings.WEATHER_API_URL}/history.json"
            params = {"key": settings.WEATHER_API_KEY, "q": city, "dt": date}
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

weather_tool = WeatherTool()
