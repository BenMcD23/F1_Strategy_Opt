from sqlalchemy.orm import joinedload
from models import TyreDeg, Driver, Session, init_db, RacingWeekend  # Replace with your actual imports
db_engine, db_session = init_db()

# Query to fetch the session_id for 2019, Round 1, Race
race_session = db_session.query(Session).join(
    RacingWeekend, Session.weekend_id == RacingWeekend.racing_weekend_id
).filter(
    RacingWeekend.year == 2019,
    RacingWeekend.round == 1,
    Session.session_type == 'Race'
).first()

if race_session:
    race_id = race_session.session_id

    # Query to fetch TyreDeg entries for the race session
    tyre_deg_entries = db_session.query(TyreDeg).options(
        joinedload(TyreDeg.driver),  # Eager load the related Driver
        joinedload(TyreDeg.session)  # Eager load the related Session
    ).filter(
        TyreDeg.race_id == race_id
    ).all()

    # Print the results
    for entry in tyre_deg_entries:
        print(f"""
        TyreDeg ID: {entry.tyre_deg_id}
        Race ID: {entry.race_id}
        Driver: {entry.driver.driver_name} (ID: {entry.driver.driver_id})
        Tyre Type: {entry.tyre_type}
        Polynomial Coefficients:
          a (x^2): {entry.a}
          b (x): {entry.b}
          c (constant): {entry.c}
        """)
else:
    print("No race found for 2019, Round 1.")