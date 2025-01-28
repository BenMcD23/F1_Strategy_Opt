import fastf1 as ff1

# Set up cache (recommended for faster loading)
ff1.Cache.enable_cache(r'C:\Users\mcdon\OneDrive - University of Leeds\Desktop\individual-project-BenMcD23\cache')

# Load the session data for Round 5 (Miami GP)
session = ff1.get_session(2019, 5, 'R')  # R = Race
session.load()

# Get all laps for Verstappen (driver number 33)
ver_laps = session.laps.pick_driver('VER')

# Convert timedelta columns to seconds for readability
ver_laps['LapTime'] = ver_laps['LapTime'].dt.total_seconds()
ver_laps['Sector1Time'] = ver_laps['Sector1Time'].dt.total_seconds()
ver_laps['Sector2Time'] = ver_laps['Sector2Time'].dt.total_seconds()
ver_laps['Sector3Time'] = ver_laps['Sector3Time'].dt.total_seconds()

# Print the laps DataFrame with driver number
print(f"All laps for Verstappen (Driver #33) at 2023 Miami GP:")
print(ver_laps[['DriverNumber', 'LapNumber', 'LapTime', 'Compound', 'TyreLife', 'Stint', 'PitOutTime', 'PitInTime']])