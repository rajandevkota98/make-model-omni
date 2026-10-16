"""get_weather via Open-Meteo (free, no API key). Returns the same shape the model trained on:
{"ok": true, "location": "Tokyo", "day": "today", "temp_c": 18, "condition": "partly cloudy"}
"""
import json
import urllib.error
import urllib.parse
import urllib.request

# WMO weather codes → the condition words used in training (configs/augment.yaml)
WMO = {0: "sunny", 1: "clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
       **{c: "drizzly" for c in (51, 53, 55, 56, 57)},
       **{c: "rainy" for c in (61, 63, 65, 66, 67, 80, 81, 82)},
       **{c: "snowy" for c in (71, 73, 75, 77, 85, 86)},
       **{c: "stormy" for c in (95, 96, 99)}}
CALM = {"sunny", "clear", "partly cloudy", "overcast"}


def get_json(url: str, params: dict, timeout: float) -> dict:
    with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=timeout) as response:
        return json.load(response)


def condition_for(code: int, wind_kmh: float, is_day: bool, windy_kmh: float) -> str:
    condition = WMO.get(code, "cloudy")
    if condition in CALM and wind_kmh >= windy_kmh:
        return "windy"
    return "clear" if condition == "sunny" and not is_day else condition


def get_weather(args: dict, env: dict) -> dict:
    wcfg = env["cfg"]["weather"]
    name, day, timeout = args.get("location") or wcfg["default_location"], args["day"], wcfg["timeout_seconds"]
    try:
        places = get_json(wcfg["geocode_url"], {"name": name, "count": 1}, timeout).get("results") or []
        if not places:
            return {"ok": False, "error": "location_not_found"}
        place = places[0]
        data = get_json(wcfg["forecast_url"], {
            "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": 2,
            "current": "temperature_2m,weather_code,wind_speed_10m,is_day",
            "daily": "temperature_2m_max,weather_code,wind_speed_10m_max"}, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
        return {"ok": False, "error": "no_network"}
    if day == "today":
        now = data["current"]
        temp, condition = now["temperature_2m"], condition_for(now["weather_code"], now["wind_speed_10m"], bool(now["is_day"]), wcfg["windy_kmh"])
    else:
        daily = data["daily"]
        temp, condition = daily["temperature_2m_max"][1], condition_for(daily["weather_code"][1], daily["wind_speed_10m_max"][1], True, wcfg["windy_kmh"])
    return {"ok": True, "location": place["name"], "day": day, "temp_c": round(temp), "condition": condition}
