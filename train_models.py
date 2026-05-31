"""
LoanIQ — Train Real ML Models
Run this ONCE to train and save models from your CSV data.

Usage:
    cd /Users/kranthiveer/Desktop/chat_bot_loanQI
    source venv/bin/activate
    python train_models.py
"""

import csv, os, joblib, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ── Load CSV ──────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'loan_data_500_extended.csv')

rows = []
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"Loaded {len(rows)} records")

# ── Feature Engineering ───────────────────────────────────────────────────────
# Features: age, income, cibil, loan_amount, tenure_months, emi, emi_ratio, loan_to_income, loan_type_enc
loan_type_enc = LabelEncoder()
loan_types = [r['loan_type'] for r in rows]
loan_type_enc.fit(loan_types)

def build_features(row):
    income      = float(row['income'])
    cibil       = float(row['cibil'])
    loan_amount = float(row['loan_amount'])
    tenure      = float(row['tenure_months'])
    emi         = float(row['emi'])
    age         = float(row['age'])
    lt_encoded  = loan_type_enc.transform([row['loan_type']])[0]
    emi_ratio   = (emi / income * 100) if income > 0 else 0
    loan_to_inc = loan_amount / (income * 12) if income > 0 else 0
    return [age, income, cibil, loan_amount, tenure, emi, emi_ratio, loan_to_inc, lt_encoded]

X = np.array([build_features(r) for r in rows])
y = np.array([1 if r['eligible'] == 'Yes' else 0 for r in rows])

print(f"Features: {X.shape}, Class balance: Yes={y.sum()}, No={len(y)-y.sum()}")

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train Models ──────────────────────────────────────────────────────────────
models = {
    'Random Forest':   RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'SVM':             SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
    'Decision Tree':   DecisionTreeClassifier(max_depth=8, random_state=42),
    'Naive Bayes':     GaussianNB(),
}

results = {}
trained = {}

for name, model in models.items():
    # SVM and NB work better with scaled features
    if name in ('SVM', 'Naive Bayes'):
        model.fit(X_train_sc, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_sc))
        cv  = cross_val_score(model, scaler.transform(X), y, cv=5).mean()
    else:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        cv  = cross_val_score(model, X, y, cv=5).mean()

    results[name] = {'accuracy': round(acc * 100, 1), 'cv': round(cv * 100, 1)}
    trained[name] = model
    print(f"  {name}: test={acc*100:.1f}%  cv={cv*100:.1f}%")

# ── Save Everything ───────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(trained['Random Forest'], f'{SAVE_DIR}/rf.pkl')
joblib.dump(trained['SVM'],           f'{SAVE_DIR}/svm.pkl')
joblib.dump(trained['Decision Tree'], f'{SAVE_DIR}/dt.pkl')
joblib.dump(trained['Naive Bayes'],   f'{SAVE_DIR}/nb.pkl')
joblib.dump(scaler,                   f'{SAVE_DIR}/scaler.pkl')
joblib.dump(loan_type_enc,            f'{SAVE_DIR}/loan_type_enc.pkl')

# Save accuracy metadata for the HTML to display
with open(f'{SAVE_DIR}/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Models saved to ./models/")
print(json.dumps(results, indent=2))
