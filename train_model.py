import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Loading  data
df = pd.read_csv('heart.csv')

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nDataset info:")
print(df.info())

# separation of columns. .
X = df.drop('HeartDisease', axis=1) if 'HeartDisease' in df.columns else df.iloc[:, :-1]
y = df['HeartDisease'] if 'HeartDisease' in df.columns else df.iloc[:, -1]

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=False)
 
print("\nTarget variable value counts:")
print(y.value_counts())

# Spliting  data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training of  model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Evaluate of the model
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print(f"\nTrain Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Saving  model in pickle file
joblib.dump(model, 'lr_heart.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')

print("\n✅ Model, scaler, and columns saved successfully!")
