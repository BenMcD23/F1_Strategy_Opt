from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


from sklearn.model_selection import train_test_split


class OvertakingModel:
	def __init__(self, race_df):
		self.__race_df = race_df
		self.feature_names = [
			"gap",
			"tyre_diff",
			"stint_laps_diff",
			"drs_available",
			"cumulative_time",
			"sector_time",
			"pace",
			"pit"
		]
		self.__imputer = SimpleImputer(strategy='mean')  # sklearn library - for missing values
		self.__model = self.__train_overtaking_model()

	def __train_overtaking_model(self):
		# Prepare feature matrix (X) and target vector (y)
		X = self.__race_df[self.feature_names].values 
		y = self.__race_df["overtake"].values

		# Handle missing values using the imputer
		X = self.__imputer.fit_transform(X)

		# Resample the data using SMOTE
		smote = SMOTE(random_state=42)    # imblearn library
		X_resampled, y_resampled = smote.fit_resample(X, y)

		# Train the GradientBoostingClassifier
		gbc = GradientBoostingClassifier(  # sklearn library
			n_estimators=200,
			learning_rate=0.05,
			max_depth=3,
			subsample=0.8,
			random_state=42
		)

		# Calibrate for better probabilities
		model = CalibratedClassifierCV(gbc, method="sigmoid", cv=3)  # sklearn library
		model.fit(X_resampled, y_resampled)

		return model

	def extract_features(self, driver_data):
		"""
		Extracts features from the active_drivers list of dictionaries and returns a NumPy array.
		
		Args:
			active_drivers (list): List of dictionaries containing driver data.
		
		Returns:
			np.ndarray: A 2D NumPy array where each row corresponds to a driver and columns correspond to features.
		"""
		# Extract features for each driver
		features = []
		for driver in driver_data:
			driver_features = [driver[feature] for feature in self.feature_names]
			features.append(driver_features)
		
		# Convert to a NumPy array
		return np.array(features, dtype=float)

	def predict_overtake(self, data):
		"""
		Predict overtakes for a NumPy array of feature data.
		
		Args:
			data (np.ndarray): A 2D NumPy array where each row represents a sample and columns correspond to features.
		
		Returns:
			np.ndarray: Predictions for each sample.
		"""
		# Ensure all features are numeric
		data = np.array([
			[int(x) if isinstance(x, bool) else x for x in row]  # Convert booleans to integers
			for row in data
		], dtype=float)  # Ensure the array is of type float

		# Handle missing values using the same imputer used during training
		data_filled = self.__imputer.transform(data)

		# Make predictions using the trained model
		predictions = self.__model.predict(data_filled)
		return predictions

	def handle_overtake_prediction(self, driver_data):
		driver_features = self.extract_features(driver_data)
		predicted_overtakes = self.predict_overtake(driver_features)
	
		return predicted_overtakes
	
	def get_model_accuracy(self):
		# Split the data into features and target
		X_test = self.__race_df[self.feature_names].values
		y_test = self.__race_df["overtake"].values

		# Predict overtakes
		predicted_overtakes = self.predict_overtake(X_test)

		# Evaluate performance
		accuracy = accuracy_score(y_test, predicted_overtakes)  # sklearn library

		# Generate classification report
		report = classification_report(  # sklearn library
			y_test,
			predicted_overtakes,
			target_names=["No Overtake", "Overtake"]
		)
		return accuracy, report


	def train_with_test_split(self, test_size=0.2, random_state=32):
		"""
		Splits the data into training and testing sets, trains the model on the training set,
		and evaluates it on the testing set.
		
		Args:
			test_size (float): Proportion of the dataset to include in the test split.
			random_state (int): Random seed for reproducibility.
		
		Returns:
			float: Accuracy score on the test set.
			str: Classification report.
		"""
		# Prepare feature matrix (X) and target vector (y)
		X = self.__race_df[self.feature_names].values
		y = self.__race_df["overtake"].values

		# Split the data into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(  # sklearn library
			X, y, test_size=test_size, random_state=random_state, stratify=y
		)

		# Handle missing values using the imputer
		X_train = self.__imputer.fit_transform(X_train)  # sklearn library
		X_test = self.__imputer.transform(X_test)  # sklearn library

		# Resample the data using SMOTE
		smote = SMOTE(random_state=42)    # imblearn library
		X_resampled, y_resampled = smote.fit_resample(X_train, y_train)  # sklearn library

		# Train the GradientBoostingClassifier
		gbc = GradientBoostingClassifier(
			n_estimators=200,
			learning_rate=0.05,
			max_depth=3,
			subsample=0.8,
			random_state=42
		)

		# Calibrate for better probabilities
		model = CalibratedClassifierCV(gbc, method="sigmoid", cv=3)  # sklearn library
		model.fit(X_resampled, y_resampled)


		# Evaluate the model on the test set
		predictions = model.predict(X_test)
		accuracy = accuracy_score(y_test, predictions)  # sklearn library
		report = classification_report(  # sklearn library
			y_test,
			predictions,
			target_names=["No Overtake", "Overtake"]
		)

		return accuracy, report