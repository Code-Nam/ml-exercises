import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
dataset_path = 'data/raw/housing-price-train.csv'
df_raw = pd.read_csv(dataset_path)
df_selected = df_raw[['MSZoning', 'YrSold', 'GarageType', 'SalePrice']].copy()

# Impute missing values in 'GarageType' with mode
if df_selected['GarageType'].isnull().any():
  mode_value = df_selected['GarageType'].mode()[0]
  df_selected['GarageType'].fillna(mode_value, inplace=True)

# Encode categorical features
le_msz = LabelEncoder()
le_gar = LabelEncoder()
df_selected['MSZoning_encoded'] = le_msz.fit_transform(df_selected['MSZoning'].astype(str))
df_selected['GarageType_encoded'] = le_gar.fit_transform(df_selected['GarageType'].astype(str))

# Define features and target
X = df_selected[['MSZoning_encoded', 'YrSold', 'GarageType_encoded']]
y = df_selected['SalePrice']

# Convert continuous SalePrice to binary label (1 if above median)
y_class = (y > y.median()).astype(int)

# Split data for classification
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(
  X, y_class, test_size=0.2, random_state=42
)

# Train RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_cl, y_train_cl)

# Predict and evaluate
y_pred_cl = clf.predict(X_test_cl)
accuracy = accuracy_score(y_test_cl, y_pred_cl)
print(f"Accuracy on validation: {accuracy*100:.2f}%")
