import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load pre-surgery stats
df = pd.read_csv('pre_surgery_stats.csv')

# Features and target
features = [
    'pre_release_speed',
    'pre_release_spin_rate',
    'pre_release_extension',
    'pre_zone_pct',
    'age_at_surgery'
]

X = df[features].copy()
y = df['successful_return'].copy()

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Train random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
cv_scores = cross_val_score(model, X_imputed, y, cv=5)

print("=== Model Results ===")
print(f"\nCross-validation accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")
print(f"\nTest set results:")
print(classification_report(y_test, y_pred, target_names=['Did Not Return', 'Returned']))

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('What Predicts a Successful Return from Tommy John Surgery?')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("\nChart saved!")