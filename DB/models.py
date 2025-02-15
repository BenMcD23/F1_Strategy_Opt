from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

class Season(Base):
	__tablename__ = 'seasons'
	year = Column(Integer, primary_key=True)

class Circuit(Base):
	__tablename__ = 'circuits'
	circuit_id = Column(Integer, primary_key=True, autoincrement=True)
	circuit_name = Column(Text, nullable=False)

	team_circuit_stats = relationship("TeamCircuitStats", back_populates="circuit")

class RacingWeekend(Base):
	__tablename__ = 'racing_weekends'
	racing_weekend_id = Column(Integer, primary_key=True, autoincrement=True)
	year = Column(Integer, ForeignKey('seasons.year'), nullable=False)
	round = Column(Integer, nullable=False)
	circuit_id = Column(Integer, ForeignKey('circuits.circuit_id'), nullable=False)

	season = relationship("Season")
	circuit = relationship("Circuit")
	sessions = relationship("Session", back_populates="racing_weekend")

class Session(Base):
	__tablename__ = 'sessions'
	session_id = Column(Integer, primary_key=True, autoincrement=True)
	weekend_id = Column(Integer, ForeignKey('racing_weekends.racing_weekend_id'), nullable=False)
	session_type = Column(Text, nullable=False)
	wet = Column(Boolean, nullable=False)

	racing_weekend = relationship("RacingWeekend", back_populates="sessions")
	session_results = relationship("SessionResult", back_populates="session")
	laps = relationship("Lap", back_populates="session")
	driver_team_sessions = relationship("DriverTeamSession", back_populates="session")

class Driver(Base):
	__tablename__ = 'drivers'
	driver_id = Column(Integer, primary_key=True, autoincrement=True)
	driver_num = Column(Integer, nullable=False)
	driver_name = Column(Text, nullable=False)
	driver_short = Column(Text, nullable=False)

	session_results = relationship("SessionResult", back_populates="driver")
	laps = relationship("Lap", back_populates="driver")
	driver_team_sessions = relationship("DriverTeamSession", back_populates="driver")

class SessionResult(Base):
	__tablename__ = 'session_results'
	session_result_id = Column(Integer, primary_key=True, autoincrement=True)
	session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
	position = Column(Integer, nullable=False)
	result_classified_pos = Column(Text, nullable=True)
	grid_pos = Column(Integer, nullable=True)
	end_status = Column(Text, nullable=True)

	session = relationship("Session", back_populates="session_results")
	driver = relationship("Driver", back_populates="session_results")

class Lap(Base):
	__tablename__ = 'laps'
	lap_id = Column(Integer, primary_key=True, autoincrement=True)
	session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
	lap_num = Column(Integer, nullable=False)
	stint_num = Column(Integer, nullable=True)
	stint_lap = Column(Integer, nullable=True)
	lap_time = Column(Float, nullable=True)
	s1_time = Column(Float, nullable=True)
	s2_time = Column(Float, nullable=True)
	s3_time = Column(Float, nullable=True)
	position = Column(Integer, nullable=True)
	tyre_type = Column(Integer, nullable=False)
	tyre_laps = Column(Integer, nullable=True)
	pit = Column(Boolean, nullable=False)
	track_status = Column(Integer, nullable=True)
	rainfall = Column(Boolean, nullable=False)

	session = relationship("Session", back_populates="laps")
	driver = relationship("Driver", back_populates="laps")
	pit_stop = relationship("PitStop", back_populates="lap")


class DriverTeamSession(Base):
	__tablename__ = 'driver_team_session'
	DTS_id = Column(Integer, primary_key=True, autoincrement=True)
	team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
	session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)

	team = relationship("Team", back_populates="driver_team_sessions")
	session = relationship("Session", back_populates="driver_team_sessions")
	driver = relationship("Driver", back_populates="driver_team_sessions")

class Team(Base):
	__tablename__ = 'teams'
	team_id = Column(Integer, primary_key=True, autoincrement=True)
	team_name = Column(Text, nullable=False)
	TeamColor = Column(Text, nullable=False)

	driver_team_sessions = relationship("DriverTeamSession", back_populates="team")
	team_circuit_stats = relationship("TeamCircuitStats", back_populates="team")

class TeamCircuitStats(Base):
	__tablename__ = 'team_circuit_stats'
	tcs_id = Column(Integer, primary_key=True, autoincrement=True)
	circuit_id = Column(Integer, ForeignKey('circuits.circuit_id'), nullable=False)
	team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
	avg_pit_time = Column(Float, nullable=False)

	circuit = relationship("Circuit", back_populates="team_circuit_stats")
	team = relationship("Team", back_populates="team_circuit_stats")

class PitStop(Base):
	__tablename__ = 'pit_stops'
	pit_id = Column(Integer, primary_key=True, autoincrement=True)
	lap_id = Column(Integer, ForeignKey('laps.lap_id'), nullable=False)
	pit_time = Column(Float, nullable=False)

	lap = relationship("Lap", back_populates="pit_stop")

def init_db():
	engine = create_engine('sqlite:////home/ben/Individual_Project/DB/f1_data_V3.db')
	Session = sessionmaker(bind=engine)
	Base.metadata.create_all(engine)
	return engine, Session()