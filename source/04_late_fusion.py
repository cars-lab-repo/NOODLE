import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss
from sklearn.model_selection import train_test_split


graph_data = pd.read_csv('actual_graph_22_16features.csv')
X_graph = graph_data.iloc[:, :-1].values
y_graph = graph_data['Trojan'].values


tabular_data = pd.read_csv('actual_table_22_16features.csv')
X_tabular = tabular_data.iloc[:, :-1].values
y_tabular = tabular_data['Trojan'].values

X_graph = np.expand_dims(X_graph, axis=2)


X_graph_train, X_graph_test, y_graph_train, y_graph_test = train_test_split(X_graph, y_graph, test_size=0.2, random_state=42)
X_tabular_train, X_tabular_test, y_tabular_train, y_tabular_test = train_test_split(X_tabular, y_tabular, test_size=0.2, random_state=42)


def create_graph_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_tabular_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


graph_model = create_graph_model(X_graph_train.shape[1:])
graph_model.fit(X_graph_train, y_graph_train, epochs=100, batch_size=5, verbose=1)


tabular_model = create_tabular_model(X_tabular_train.shape[1:])
tabular_model.fit(X_tabular_train, y_tabular_train, epochs=100, batch_size=5, verbose=1)


graph_predictions = graph_model.predict(X_graph_test)
tabular_predictions = tabular_model.predict(X_tabular_test)


combined_predictions = (graph_predictions + tabular_predictions) / 2


combined_predictions_binary = (combined_predictions > 0.5).astype(int)


accuracy = accuracy_score(y_graph_test, combined_predictions_binary)
confusion = confusion_matrix(y_graph_test, combined_predictions_binary)
brier_score = brier_score_loss(y_graph_test, combined_predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:\n{confusion}')
print(f'Brier Score: {brier_score:.4f}')
