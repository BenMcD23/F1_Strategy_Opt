import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from DB.models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap


def create_dataframe():
    # initialize db connection and session
    db_engine, db_session = init_db()

    # query data from the database
    query = db_session.query(
        RacingWeekend.year,
        RacingWeekend.round,
        Circuit.circuit_name,
        Driver.driver_name,
        Driver.driver_short,
        Lap.lap_num,
        Lap.lap_time,
        Lap.tyre,
        Lap.pit,
        Session.session_type
    ).join(RacingWeekend.circuit) \
     .join(RacingWeekend.sessions) \
     .join(Session.laps) \
     .join(Lap.driver) \
     .join(RacingWeekend.season) \
     .all()

    # convert result to list of dicts
    data = []
    for row in query:
        data.append({
            'year': row.year,
            'round': row.round,
            'circuit_name': row.circuit_name,
            'driver_name': row.driver_name,
            'driver_short': row.driver_short,
            'lap_num': row.lap_num,
            'lap_time': row.lap_time,
            'tyre': row.tyre,
            'pit': row.pit,
            'session_type': row.session_type
        })

    # create dataframe
    df = pd.DataFrame(data)

    # encode 'tyre' and 'pit' as categories
    df['tyre'] = df['tyre'].astype('category')
    df['pit'] = df['pit'].astype('category')

    # one-hot encode categorical variables
    df = pd.get_dummies(df, columns=['circuit_name', 'driver_name', 'driver_short', 'session_type'], drop_first=True)

    return df


# Create the dataframe and print it
df = create_dataframe()

# Split data into training (2019, 2020, 2021) and testing (2022)
train_data = df[df['year'].isin([2019, 2020, 2021])]
test_data = df[df['year'] == 2022]

# Define features (X) and targets (y) for training
X_train = train_data.drop(columns=['pit', 'tyre'])
y_train = train_data[['pit', 'tyre']]

# Define features (X) and targets (y) for testing
X_test = test_data.drop(columns=['pit', 'tyre'])
y_test = test_data[['pit', 'tyre']]

# Train the MultiOutput Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
multi_output_rf = MultiOutputClassifier(rf_classifier)
multi_output_rf.fit(X_train, y_train)

# Predict on the entire 2022 season
y_pred_test = multi_output_rf.predict(X_test)

# Add predictions to the test data
test_data['predicted_pit'] = y_pred_test[:, 0]
test_data['predicted_tyre'] = y_pred_test[:, 1]

# Evaluate the model
print("=== Model Evaluation ===")

# Accuracy for both pit and tyre
accuracy_pit = accuracy_score(y_test['pit'], y_pred_test[:, 0])
accuracy_tyre = accuracy_score(y_test['tyre'], y_pred_test[:, 1])
print(f"Accuracy (Pit): {accuracy_pit:.2f}")
print(f"Accuracy (Tyre): {accuracy_tyre:.2f}")

# Confusion Matrix for both pit and tyre
conf_matrix_pit = confusion_matrix(y_test['pit'], y_pred_test[:, 0])
conf_matrix_tyre = confusion_matrix(y_test['tyre'], y_pred_test[:, 1])

print("\nConfusion Matrix (Pit):")
print(conf_matrix_pit)

print("\nConfusion Matrix (Tyre):")
print(conf_matrix_tyre)

# Classification Report for both pit and tyre
class_report_pit = classification_report(y_test['pit'], y_pred_test[:, 0])
class_report_tyre = classification_report(y_test['tyre'], y_pred_test[:, 1])

print("\nClassification Report (Pit):")
print(class_report_pit)

print("\nClassification Report (Tyre):")
print(class_report_tyre)


