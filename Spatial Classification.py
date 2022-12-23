# -*- coding: utf-8 -*-

# DECISION TREE model

# 1- Function for Downloading Dataset

import requests
import numpy as np

def load_satellite_data():
    # Load the data from a URL
    url = 'https://s3.amazonaws.com/aerial-imagery-sample/Aerial_Imagery_Sample.zip'
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall()

    # Load the image files as numpy arrays
    image_files = glob.glob('*.tif')
    images = [np.array(Image.open(f)) for f in image_files]

    # Load the labels from a CSV file
    labels = pd.read_csv('labels.csv')['label'].values

    return images, labels

# 2- Function for Building a Model 

from sklearn.ensemble import RandomForestClassifier

def build_model():
    # Create the model
    model = RandomForestClassifier()

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

# 3- Spliting The Data

from sklearn.model_selection import train_test_split

# Load the dataset
X, y = load_satellite_data()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""step 04: Preprocessing model"""

# Preprocess the data
X_train, X_test = preprocess_data(X_train, X_test)

"""STEP 05: Training and testing model"""

# Build the model
model = build_model()

# Fit the model on the training data
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=10,
                    validation_data=(X_test, y_test))

# Make predictions on the test data
y_pred = model.predict(X_test)

"""STEP 06: Evaluating the loss and accuracy of model"""

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.2f}')

from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print(cm)
