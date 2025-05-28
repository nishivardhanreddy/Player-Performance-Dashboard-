import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# 1. Load Data
df = pd.read_csv("cricket_data.csv")

# 2. Convert key columns to numeric
batting_cols = ["Batting_Average", "Batting_Strike_Rate", "Matches_Batted", "Centuries", "Half_Centuries"]
df[batting_cols] = df[batting_cols].apply(pd.to_numeric, errors="coerce")

# 3. Drop missing values
df.dropna(subset=batting_cols, inplace=True)

# 4. Add "Form_Status": In Form = 1 if Avg ≥ 35
df["Form_Status"] = df["Batting_Average"].apply(lambda x: 1 if x >= 35 else 0)

# 5. Feature matrix & target
X = df[batting_cols]
y = df["Form_Status"]

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 9. Save model
joblib.dump(model, "models/form_model.pkl")
print("✅ Model saved to models/form_model.pkl")
