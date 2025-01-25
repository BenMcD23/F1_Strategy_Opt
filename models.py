from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

# circuits table
class Circuit(Base):
    __tablename__ = 'circuits'
    
    circuit_id = Column(Integer, primary_key=True, autoincrement=True)
    circuit_name = Column(String, nullable=False)
    
    # relationship to racing_weekends
    racing_weekends = relationship("RacingWeekend", back_populates="circuit")


# seasons table
class Season(Base):
    __tablename__ = 'seasons'
    
    year = Column(Integer, primary_key=True)
    
    # relationship to racing_weekends
    racing_weekends = relationship("RacingWeekend", back_populates="season")


# racing weekends table
class RacingWeekend(Base):
    __tablename__ = 'racing_weekends'
    
    racing_weekend_id = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, ForeignKey('seasons.year'), nullable=False)
    round = Column(Integer, nullable=False)
    circuit_id = Column(Integer, ForeignKey('circuits.circuit_id'), nullable=False)

    # relationships
    season = relationship("Season", back_populates="racing_weekends")
    circuit = relationship("Circuit", back_populates="racing_weekends")
    sessions = relationship("Session", back_populates="racing_weekend")


# drivers Table
class Driver(Base):
    __tablename__ = 'drivers'
    
    driver_id = Column(Integer, primary_key=True, autoincrement=True)
    driver_num = Column(Integer, nullable=False)
    driver_name = Column(String, nullable=False)
    driver_short = Column(String, nullable=False)
    
    # relationships
    session_results = relationship("SessionResult", back_populates="driver")
    laps = relationship("Lap", back_populates="driver")
    tyre_deg = relationship("TyreDeg", back_populates="driver")


# sessions Table
class Session(Base):
    __tablename__ = 'sessions'
    
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    weekend_id = Column(Integer, ForeignKey('racing_weekends.racing_weekend_id'), nullable=False)
    session_type = Column(String, nullable=False)
    
    # relationship
    racing_weekend = relationship("RacingWeekend", back_populates="sessions")
    session_results = relationship("SessionResult", back_populates="session")
    laps = relationship("Lap", back_populates="session")
    tyre_deg = relationship("TyreDeg", back_populates="session")



# session results Table
class SessionResult(Base):
    __tablename__ = 'session_results'
    
    session_result_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
    driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
    position = Column(Integer, nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="session_results")
    driver = relationship("Driver", back_populates="session_results")


# laps table
class Lap(Base):
    __tablename__ = 'laps'
    
    lap_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
    driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
    lap_num = Column(Integer, nullable=True)
    lap_time = Column(Float, nullable=False)
    position = Column(Integer, nullable=True)
    tyre = Column(Integer, nullable=True)
    tyre_laps = Column(Integer, nullable=True)
    pit = Column(Boolean, nullable=False)  # 0 = No pit, 1 = Pit
    rainfall = Column(Boolean, nullable=True)  # 0 = No rain, 1 = rain
    track_temp = Column(Float, nullable=True)
    
    # relationships
    session = relationship("Session", back_populates="laps")
    driver = relationship("Driver", back_populates="laps")


class TyreDeg(Base):
    __tablename__ = 'tyre_deg'
    
    tyre_deg_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey('sessions.session_id'), nullable=False)
    driver_id = Column(Integer, ForeignKey('drivers.driver_id'), nullable=False)
    tyre_type = Column(Integer, nullable=False)
    a = Column(Float, nullable=True)
    b = Column(Float, nullable=True)
    c = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="tyre_deg")
    driver = relationship("Driver", back_populates="tyre_deg")


def init_db():
    # create db if isnt already
    engine = create_engine('sqlite:///f1_data_V3.db')
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    session = Session()

    return engine, session