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


def create_convolutional_layers(input_tensor):
    conv1 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(input_tensor)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    return pool1


def create_dense_layers(input_tensor):
    dense1 = tf.keras.layers.Dense(64, activation='relu')(input_tensor)
    return dense1


graph_input = tf.keras.layers.Input(shape=(X_graph.shape[1], X_graph.shape[2]))
graph_conv_layers = create_convolutional_layers(graph_input)

graph_flat = tf.keras.layers.Flatten()(graph_conv_layers)

tabular_input = tf.keras.layers.Input(shape=(X_tabular.shape[1],))
tabular_dense_layers = create_dense_layers(tabular_input)

concatenated = tf.keras.layers.concatenate([graph_flat, tabular_dense_layers])


x = tf.keras.layers.Dense(128, activation='relu')(concatenated)
x = tf.keras.layers.Dropout(0.5)(x) 
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=[graph_input, tabular_input], outputs=output)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit([X_graph_train, X_tabular_train], y_graph_train, epochs=100, batch_size=5, verbose=1)


y_pred = model.predict([X_graph_test, X_tabular_test])
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_graph_test, y_pred_binary)
confusion = confusion_matrix(y_graph_test, y_pred_binary)
brier_score = brier_score_loss(y_graph_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:\n{confusion}')
print(f'Brier Score: {brier_score:.4f}')
