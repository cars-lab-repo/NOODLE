import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# Load tabular data
tabular_data = pd.read_csv('tabular_dataset.csv')

# Process JSON data
json_features = []

for i in range(10):
    with open(f'synthetic_{i}.json', 'r') as f:
        data = json.load(f)
        features = []

        for key in data['features']:
            features.extend(data['features'][key][0])

        json_features.append(features)

# Convert to numpy arrays
X_tabular = tabular_data[['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']].values
X_json = np.array(json_features)

# Standardize tabular features
scaler = StandardScaler()
X_tabular = scaler.fit_transform(X_tabular)

# Split data into train and test sets
X_tabular_train, X_tabular_test, X_json_train, X_json_test, y_train, y_test = train_test_split(
    X_tabular, X_json, tabular_data['Label'], test_size=0.2, random_state=42
)

# Define the tabular input
tabular_input = Input(shape=(5,), name='tabular_input')
x = Dense(16, activation='relu')(tabular_input)

# Define the JSON input
json_input = Input(shape=(len(X_json_train[0]),), name='json_input')
y = Dense(16, activation='relu')(json_input)

# Concatenate both inputs
merged = Concatenate()([x, y])
output = Dense(1, activation='sigmoid')(merged)

# Define the model
model = Model(inputs=[tabular_input, json_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_tabular_train, X_json_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate([X_tabular_test, X_json_test], y_test)
print(f'Accuracy: {accuracy}')
