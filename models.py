from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

class Season(Base):
	__tablename__ = 'seasons'
	year = Column(Integer, primary_key=True)
	racing_weekend = relationship("RacingWeekend", back_populates="season")

class Circuit(Base):
	__tablename__ = 'circuits'
	circuit_id = Column(Integer, primary_key=True, autoincrement=True)
	circuit_name = Column(Text, nullable=False)
	racing_weekend = relationship("RacingWeekend", back_populates="circuit")
	team_circuit_stats = relationship("TeamCircuitStats", back_populates="circuit")

class RacingWeekend(Base):
	__tablename__ = 'racing_weekends'
	racing_weekend_id = Column(Integer, primary_key=True, autoincrement=True)
	year = Column(Integer, ForeignKey('seasons.year'), nullable=False)
	round = Column(Integer, nullable=False)
	circut_id = Column(Integer, ForeignKey('circuits.circuit_id'), nullable=False)
	
	season = relationship("Season", back_populates="racing_weekend")
	circuit = relationship("Circuit", back_populates="racing_weekend")
	session = relationship("Session", back_populates="racing_weekend")

class Driver(Base):
	__tablename__ = 'drivers'
	driver_id = Column(Integer, primary_key=True, autoincrement=True)
	driver_num = Column(Integer, nullable=False)
	driver_name = Column(Text, nullable=False)
	driver_short = Column(Text, nullable=False)
	
	session_result = relationship("SessionResult", back_populates="driver")
	lap = relationship("Lap", back_populates="driver")
	driver_team_session = relationship("DriverTeamSession", back_populates="driver")
	tyre_race_data = relationship("TyreRaceData", back_populates="driver")

class Session(Base):
	__tablename__ = 'sessions'
	session_id = Column(Integer, primary_key=True, autoincrement=True)
	weekend_id = Column(Integer, ForeignKey('racing_weekends.racing_weekend_id'), nullable=False)
	session_type = Column(Text, nullable=False)
	wet = Column(Boolean, nullable=True)
	
	racing_weekend = relationship("RacingWeekend", back_populates="session")
	session_result = relationship("SessionResult", back_populates="session")
	lap = relationship("Lap", back_populates="session")
	driver_team_session = relationship("DriverTeamSession", back_populates="session")
	tyre_race_data = relationship("TyreRaceData", back_populates="session")

class SessionResult(Base):
	__tablename__ = 'session_results'
	session_result_id = Column(Integer, primary_key=True, autoincrement=True)
	session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
	position = Column(Integer, nullable=False)
	
	session = relationship("Session", back_populates="session_result")
	driver = relationship("Driver", back_populates="session_result")

class Lap(Base):
	__tablename__ = 'laps'
	lap_id = Column(Integer, primary_key=True, autoincrement=True)
	session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
	lap_num = Column(Integer, nullable=False)
	stint_num = Column(Integer, nullable=True)
	lap_time = Column(Float, nullable=False)
	position = Column(Integer, nullable=True)
	tyre = Column(Integer, nullable=False)
	tyre_laps = Column(Integer, nullable=True)
	pit = Column(Boolean, nullable=False)
	rainfall = Column(Boolean, nullable=False)
	
	session = relationship("Session", back_populates="lap")
	driver = relationship("Driver", back_populates="lap")

class TyreRaceData(Base):
	__tablename__ = 'tyre_race_data'
	race_data_id = Column(Integer, primary_key=True, autoincrement=True)
	race_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
	tyre_type = Column(Integer, nullable=False)
	a = Column(Float, nullable=False)
	b = Column(Float, nullable=False)
	c = Column(Float, nullable=False)
	
	session = relationship("Session", back_populates="tyre_race_data")
	driver = relationship("Driver", back_populates="tyre_race_data")

class DriverTeamSession(Base):
	__tablename__ = 'driver_team_session'
	DTS_id = Column(Integer, primary_key=True, autoincrement=True)
	team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
	session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
	driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
	
	team = relationship("Team", back_populates="driver_team_session")
	session = relationship("Session", back_populates="driver_team_session")
	driver = relationship("Driver", back_populates="driver_team_session")

class Team(Base):
	__tablename__ = 'teams'
	team_id = Column(Integer, primary_key=True, autoincrement=True)
	team_name = Column(Text, nullable=False)
	TeamColor = Column(Text, nullable=False)
	
	driver_team_session = relationship("DriverTeamSession", back_populates="team")
	team_circuit_stats = relationship("TeamCircuitStats", back_populates="team")

class TeamCircuitStats(Base):
	__tablename__ = 'team_circuit_stats'
	pit_id = Column(Integer, primary_key=True, autoincrement=True)
	circuit_id = Column(Integer, ForeignKey('circuits.circuit_id'), nullable=False)
	team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
	pit_time = Column(Float, nullable=False)
	quali_to_race_percent_diff = Column(Float, nullable=False)
	
	circuit = relationship("Circuit", back_populates="team_circuit_stats")
	team = relationship("Team", back_populates="team_circuit_stats")

def init_db():
	engine = create_engine('sqlite:///f1_data_V2.db')
	Session = sessionmaker(bind=engine)
	Base.metadata.create_all(engine)
	return engine, Session()