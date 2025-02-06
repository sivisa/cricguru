# cricguru
Cricket team prediction base code
mport numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Example of loading and combining datasets

# Load datasets
player_data = pd.read_csv('player_performance.csv')  # Features related to players (batting, bowling stats)
stadium_data = pd.read_csv('stadium_conditions.csv')  # Features related to stadiums
environment_data = pd.read_csv('environment_factors.csv')  # Weather and environmental factors

# Combine datasets (by player and match IDs, assuming you have a common key for all datasets)
data = pd.concat([player_data, stadium_data, environment_data], axis=1)

# Split features and target
X = data.drop(columns=['is_selected'])  # Drop the target variable
y = data['is_selected']  # 1 for selected, 0 for not selected (as a classification problem)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (for numerical data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Input layers for player performance, stadium conditions, and environmental factors
player_input = Input(shape=(X_train_scaled.shape[1],), name='player_input')
stadium_input = Input(shape=(stadium_data.shape[1],), name='stadium_input')
environment_input = Input(shape=(environment_data.shape[1],), name='environment_input')

# Player Performance Sub-network
x1 = Dense(64, activation='relu')(player_input)
x1 = Dropout(0.2)(x1)
x1 = BatchNormalization()(x1)

# Stadium Conditions Sub-network
x2 = Dense(32, activation='relu')(stadium_input)
x2 = Dropout(0.2)(x2)
x2 = BatchNormalization()(x2)

# Environmental Factors Sub-network
x3 = Dense(32, activation='relu')(environment_input)
x3 = Dropout(0.2)(x3)
x3 = BatchNormalization()(x3)

# Concatenate the outputs of the three sub-networks
merged = Concatenate()([x1, x2, x3])

# Decision-making layers (fully connected)
x = Dense(64, activation='relu')(merged)
x = Dropout(0.3)(x)
x = BatchNormalization()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)

# Final output layer for binary classification (selected/not selected)
output = Dense(1, activation='sigmoid', name='output')(x)

# Model definition
model = Model(inputs=[player_input, stadium_input, environment_input], outputs=[output])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Fit the model
history = model.fit(
    [X_train_scaled, stadium_data, environment_data], y_train,
    validation_data=([X_test_scaled, stadium_data, environment_data], y_test),
    epochs=50,
    batch_size=32
)

# Evaluate the model performance
loss, accuracy = model.evaluate([X_test_scaled, stadium_data, environment_data], y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predict probabilities
y_pred_probs = model.predict([X_test_scaled, stadium_data, environment_data])
# Convert probabilities to class labels (1 or 0)
y_pred = (y_pred_probs > 0.5).astype(int)

# Evaluate on test set
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
