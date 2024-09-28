import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, log_loss

# Load the balanced dataset
csv_balanced_concat_dataset_file = 'path/to/your/balanced_dataset.csv'
data = pd.read_csv(csv_balanced_concat_dataset_file)

# Preprocess the data
# Assuming 'race' is the target variable and the rest are features
X = data.drop(columns=['race'])
y = data['race']

# Encode categorical variables
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
logloss = log_loss(y_val, y_prob)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Log Loss: {logloss:.4f}')
