import gradio as gr
import requests
import datetime

def predict(date, location, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir,
            wind_gust_speed, wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm,
            humidity_9am, humidity_3pm, pressure_9am, pressure_3pm, cloud_9am, cloud_3pm,
            temp_9am, temp_3pm, rain_today):
    
    print("In Prediction")

    formatted_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
    response = requests.post("http://localhost:8800/predict/", json={
        "Date": str(formatted_date),
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today
    })
    print(response.json())
    return response.json()['prediction_str']

wind_directions = ["E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW", "S", "SE", "SSE", "SSW", "SW", "W", "WNW", "WSW"]

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.DateTime(label="Date"),
        gr.Dropdown(choices=[
            "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", "Ballarat",
            "Bendigo", "Brisbane", "Cairns", "Canberra", "Cobar", "CoffsHarbour",
            "Dartmoor", "Darwin", "GoldCoast", "Hobart", "Katherine", "Launceston",
            "Melbourne", "MelbourneAirport", "Mildura", "Moree", "MountGambier",
            "MountGinini", "Newcastle", "Nhil", "NorahHead", "NorfolkIsland",
            "Nuriootpa", "PearceRAAF", "Penrith", "Perth", "PerthAirport", "Portland",
            "Richmond", "Sale", "SalmonGums", "Sydney", "SydneyAirport", "Townsville",
            "Tuggeranong", "Uluru", "WaggaWagga", "Walpole", "Watsonia", "Williamtown",
            "Witchcliffe", "Wollongong", "Woomera"
        ], label="Location"),
        gr.Number(label="Min Temp"),
        gr.Number(label="Max Temp"),
        gr.Number(label="Rainfall"),
        gr.Number(label="Evaporation"),
        gr.Number(label="Sunshine", minimum=0, maximum=24),
        gr.Dropdown(choices=wind_directions, label="Wind Gust Direction"),
        gr.Number(label="Wind Gust Speed"),
        gr.Dropdown(choices=wind_directions, label="Wind Direction 9am"),
        gr.Dropdown(choices=wind_directions, label="Wind Direction 3pm"),
        gr.Number(label="Wind Speed 9am"),
        gr.Number(label="Wind Speed 3pm"),
        gr.Number(label="Humidity 9am"),
        gr.Number(label="Humidity 3pm"),
        gr.Number(label="Pressure 9am"),
        gr.Number(label="Pressure 3pm"),
        gr.Number(label="Cloud Cover 9am"),
        gr.Number(label="Cloud Cover 3pm"),
        gr.Number(label="Temp 9am"),
        gr.Number(label="Temp 3pm"),
        gr.Number(label="Rain Today", minimum=0, maximum=1)
    ],
    outputs="text",
    title="Rain Prediction Model",
    description="Introduce los datos del tiempo de hoy para predecir si lloverá mañana."
)
iface.launch()
