import pickle  # Used for serializing and deserializing Python objects (e.g., data, models)
import numpy as np  # Library for handling arrays and performing numerical computations
from sklearn.ensemble import RandomForestClassifier  # Machine learning model: Random Forest Classifier
from sklearn.model_selection import train_test_split  # Used for splitting the dataset into training and testing sets
from sklearn.metrics import accuracy_score  # Metric to evaluate model performance (accuracy)

# Load the dataset from a pickle file ('rb' means reading the file in binary mode)
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert the data and labels into NumPy arrays for easier manipulation in machine learning
data = np.asarray(data_dict['data'])  # Feature data (e.g., hand landmarks)
labels = np.asarray(data_dict['labels'])  # Labels (class indices for each sample)

# Split the dataset into training and testing sets
# 80% of the data will be used for training, and 20% will be used for testing
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model using the training data (fit the model to the data)
model.fit(x_train, y_train)

# Use the trained model to predict labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy of the model's predictions
score = accuracy_score(y_predict, y_test)

# Print the accuracy score as a percentage, formatted to two decimal places
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a file using pickle ('wb' means writing the file in binary mode)
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
