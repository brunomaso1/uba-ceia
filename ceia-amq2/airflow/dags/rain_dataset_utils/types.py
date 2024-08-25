# Este archivo contiene todos las estructuras estáticas.

COLUMNS_TYPES = {
    "cat_columns": ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"],
    "bool_columns": ["RainToday"],
    "date_columns": ["Date"],
    "cont_columns": [
        "MinTemp",
        "MaxTemp",
        "Rainfall",
        "Evaporation",
        "Sunshine",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Cloud9am",
        "Cloud3pm",
        "Temp9am",
        "Temp3pm",
    ],
}

WIND_DIRS = [
    "E",
    "ENE",
    "NE",
    "NNE",
    "N",
    "NNW",
    "NW",
    "WNW",
    "W",
    "WSW",
    "SW",
    "SSW",
    "S",
    "SSE",
    "SE",
    "ESE",
]
