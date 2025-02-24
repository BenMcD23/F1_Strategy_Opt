from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from custom_exceptions import SessionNotFoundError

from DB.models import Circuit, RacingWeekend, Driver, Session, SessionResult

class DatabaseOperations:
	""" 
	This class handles all operations with the database for a given race
	"""
	_Session = None

	# classmethod so the session isnt recreated every time we create a new instance of the class
	@classmethod
	def init_db(cls):
		if cls._Session is None:
			Base = declarative_base()
			cls._engine = create_engine('sqlite:////home/ben/Individual_Project/DB/f1_data_V4.db')
			cls._Session = sessionmaker(bind=cls._engine)
			Base.metadata.create_all(cls._engine)


	def __init__(self, year, circuit):
		"""Constructor

		Args:
			year (int): the year the race happened in. Could be 2022-2024 inclusive
			circuit (string): the name of the circuit of the race we want
		"""
		# Init db if it isnt already
		if DatabaseOperations._Session is None:
			DatabaseOperations.init_db()
		self.year = year
		self.circuit = circuit
		self.db_session = DatabaseOperations._Session()
		self.race_session_db = self._get_session("Race")
		self.race_session_results_db = self._get_session_results_db()
		self.quali_session_db = self._get_session("Qualifying")
	
	def _get_session(self, session_type):
		"""Gets the database session for the given session type, such as Race or Qualifying

		Args:
			session_type (string): the type of session the user wants, such as Race or Qualifying

		Raises:
			SessionNotFoundError: A custom exception for if the session doesn't exist in the database

		Returns:
			sqlalchemy object: The object for the session in the sqlalchemy database
		"""
		session = (self.db_session.query(Session)
			.join(RacingWeekend, Session.weekend_id == RacingWeekend.racing_weekend_id)
			.join(Circuit, RacingWeekend.circuit_id == Circuit.circuit_id)
			.filter(
				RacingWeekend.year == self.year,
				Circuit.circuit_name == self.circuit,
				Session.session_type == session_type
			)
			.first())

		if not session:
			raise SessionNotFoundError(f"No {session_type} session found for year {self.year} at circuit {self.circuit}")

		return session


	def _get_session_results_db(self):
		"""Retrieves the results of the race session from the database.

		Returns:
			list: A list of tuples containing the following information for each driver:
				- grid_pos (int): The starting grid position of the driver
				- driver_num (int): The drivers number (unique to every driver)
				- end_status (str): The status of the driver at the end of the race (Either finished, +Number lap or reason why not finished)

		Raises:
			SessionNotFoundError: A custom exception for if the session doesn't exist in the database
		"""
			
		session_results = (
			self.db_session.query(SessionResult.grid_pos, Driver.driver_num, SessionResult.end_status)
			.join(Session, Session.session_id == SessionResult.session_id)
			.join(Driver, Driver.driver_id == SessionResult.driver_id)
			.filter(SessionResult.session_id == self.race_session_db.session_id)
			.all()
		)

		if not session_results:
			raise SessionNotFoundError(f"No race results found for year {self.year} at circuit {self.circuit}")

		return session_results