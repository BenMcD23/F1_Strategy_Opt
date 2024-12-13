import fastf1 as ff1
from fastf1 import plotting
from fastf1.core import DataNotLoadedError
import logging
logging.getLogger('fastf1').setLevel(logging.WARNING)

import pandas as pd
from datetime import datetime
import time

# ff1.Cache.set_disabled()
ff1.Cache.enable_cache(r'D:\FastF1_Data\cache')



# Load the session (example: 2023 Bahrain Grand Prix, Race)
session = ff1.get_session(2019, 'China', 'R')
session.load()

# Get weather data


laps = session.laps
print(laps)

weather_data = session.laps.get_weather_data()
print(weather_data)

rainfall_data = weather_data[weather_data['Rainfall'] == True]

# Print the filtered data
print(rainfall_data)