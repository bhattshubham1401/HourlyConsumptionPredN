import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM


class ModelH:
    def __init__(self, train_data, val_data, test_data):
        self.trained_model = None
        self.sequence_length = None
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def create_sequences(self, data):
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length])

        return np.array(sequences), np.array(targets)

    def buildingModel(self):
        self.sequence_length = 24
        num_features = 1
        for location_id in self.train_data['location_id'].unique():
            train_location_data = self.train_data[self.train_data['location_id'] == location_id]['hourly_diff'].values
            val_location_data = self.val_data[self.val_data['location_id'] == location_id]['hourly_diff'].values
            test_location_data = self.test_data[self.test_data['location_id'] == location_id]['hourly_diff'].values

            train_sequences, train_targets = self.create_sequences(train_location_data)
            val_sequences, val_targets = self.create_sequences(val_location_data)
            test_sequences, test_targets = self.create_sequences(test_location_data)

            # Create and train the model
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(self.sequence_length, num_features)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(train_sequences, train_targets, validation_data=(val_sequences, val_targets), epochs=10,
                      batch_size=32)
            loss = model.evaluate(test_sequences, test_targets)
            print(f"Location ID: {location_id}, Test Loss: {loss}")

            # Train the model
            model.fit(train_sequences, train_targets, validation_data=(val_sequences, val_targets), epochs=10,
                      batch_size=32)

            # Evaluate the model
            loss = model.evaluate(test_sequences, test_targets)
            print(f"Location ID: {location_id}, Test Loss: {loss}")
        self.trained_model = model

    def generate_predictions(self, test_data, scaler):
        predictions = {}

        for location_id in self.test_data['location_id'].unique():
            location_test_data = self.test_data[self.test_data['location_id'] == location_id]
            last_creation_time = location_test_data['creation_time'].iloc[-1]

            input_sequence = location_test_data['hourly_diff_scaled'].values[-self.sequence_length:]
            pd.Timestamp(location_test_data.index[-1])

            predicted_values_actual = []

            for i in range(24):
                input_sequence_reshaped = input_sequence.reshape(1, self.sequence_length, 1)
                predicted_value_scaled = self.trained_model.predict(input_sequence_reshaped)[0][0]
                predicted_value_actual = scaler.inverse_transform(np.array([[predicted_value_scaled]]))[0][0]
                predicted_values_actual.append(predicted_value_actual)

                # Update input_sequence for the next iteration
                input_sequence = np.append(input_sequence[1:], predicted_value_scaled)

            future_datetimes = [last_creation_time + pd.Timedelta(hours=hour) for hour in range(1, 25)]

            # Calculate RMSE and MAE scores
            actual_values = location_test_data['hourly_diff'].values[-24:]
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values_actual))
            mae = mean_absolute_error(actual_values, predicted_values_actual)

            # Include predicted data, RMSE, MAE, and last_creation_time in the output
            predictions[location_id] = {
                'predictions': list(zip(future_datetimes, predicted_values_actual)),
                'rmse': rmse,
                'mae': mae,
                'last_creation_time': last_creation_time  # Include last_creation_time in the output
            }

        return predictions