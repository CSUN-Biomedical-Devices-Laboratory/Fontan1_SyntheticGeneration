import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix


######## This script loads our mixed dataset and trains a neural network against 
######## real/Fake labels. It then tests against unused entries and produces a confusion matrix


# Load dataset
data = pd.read_csv('mixed_imputed.csv')  

# Convert 'REAL'/'FAKE' into integers
data['NAME'] = data['NAME'].map({'FAKE': 0, 'REAL': 1})

# Separate features and target variable
X = data.drop('NAME', axis=1)
y = data['NAME']
232
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)

# Identify categorical and numerical columns
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Define the neural network model
def build_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=[input_shape]),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Get the input shape for the model from processed feature shape
input_shape = X_train_processed.shape[1]

# Build and compile the model
nn_model = build_model(input_shape)

# Train the model
nn_model.fit(X_train_processed, y_train, epochs=100, batch_size=32)

# Evaluate the model
loss, accuracy = nn_model.evaluate(X_test_processed, y_test)
print(f'Test accuracy: {accuracy}')

# Predictions and evaluation
y_pred = (nn_model.predict(X_test_processed) > 0.5).astype('int32')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
