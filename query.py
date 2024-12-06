from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap
import fastf1 as ff1

db_engine, db_session = init_db()


# season = db_session.query(Season).filter_by(year=2018).first()



# results of race
# for result in season.racing_weekends[0].sessions[4].session_results:
#     print(f" {result.driver.driver_short}                       {result.position}")

# adding up for total race time


# def convert_seconds_to_hhmmssmm(total_seconds):
#     total_seconds = int(total_seconds)
#     hours, remainder = divmod(total_seconds, 3600)
#     minutes, seconds = divmod(remainder, 60)
#     milliseconds = int((total_seconds - int(total_seconds)) * 1000)
#     return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


# DriversTotal = {}
# for idx, lap in enumerate(season.racing_weekends[0].sessions[4].laps):

#     if lap.driver.driver_short in DriversTotal:
#         DriversTotal[lap.driver.driver_short] += (lap.lap_time)
#     else:
#         DriversTotal.update({lap.driver.driver_short: lap.lap_time})

# print(DriversTotal)

# converted_lap_times = {driver: convert_seconds_to_hhmmssmm(time) for driver, time in DriversTotal.items()}
# print(converted_lap_times)

# # Close the session when done
# db_session.close()


data = ff1.get_session(2020, 13, "Practice 2")
data.load()

print(data)
