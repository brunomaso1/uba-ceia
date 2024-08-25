import numpy as np
import pandas as pd


def encode_cyclical_date(df, date_column="Date"):
    """
    Encodes a date column into cyclical features using sine and cosine transformations.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the date column.
    date_column (str): The name of the date column. Default is 'Date'.

    Returns:
    pandas.DataFrame: The dataframe with new 'DayCos' and 'DaySin' columns added,
                      and intermediate columns removed.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])

    # Calculate day of year
    df["DayOfYear"] = df[date_column].dt.dayofyear

    # Determine the number of days in the year for each date (taking leap years into account)
    df["DaysInYear"] = df[date_column].dt.is_leap_year.apply(
        lambda leap: 366 if leap else 365
    )

    # Convert day of the year to angle in radians, dividing by DaysInYear + 1
    df["Angle"] = 2 * np.pi * (df["DayOfYear"] - 1) / df["DaysInYear"]

    # Calculate sine and cosine features
    df["DayCos"] = np.cos(df["Angle"])
    df["DaySin"] = np.sin(df["Angle"])

    # Remove intermediate columns
    df = df.drop(columns=["DayOfYear", "DaysInYear", "Angle"])

    return df


def encode_wind_dir(df):
    dirs = [
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
    angles = np.radians(np.arange(0, 360, 22.5))
    mapping_dict = {d: a for (d, a) in zip(dirs, angles)}

    wind_dir_columns = ["WindGustDir", "WindDir9am", "WindDir3pm"]
    for column in wind_dir_columns:
        df[f"{column}Angle"] = df[column].map(mapping_dict)

        df[f"{column}Cos"] = np.cos(df[f"{column}Angle"].astype(float))
        df[f"{column}Sin"] = np.sin(df[f"{column}Angle"].astype(float))

        df = df.drop(columns=f"{column}Angle")

    return df

def encode_location(df: pd.DataFrame, gdf_locations) -> pd.DataFrame:
    return pd.merge(df, gdf_locations.drop(columns="geometry"), on="Location")