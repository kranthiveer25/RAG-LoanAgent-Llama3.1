

import os, json
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Allow requests from file:// and localhost (needed for standalone HTML)
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

@app.route('/predict/home', methods=['OPTIONS'])
@app.route('/predict/education', methods=['OPTIONS'])
@app.route('/health', methods=['OPTIONS'])
def options_handler():
    return '', 204

# ── Load saved models ─────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    rf      = joblib.load(f'{MODELS_DIR}/rf.pkl')
    svm     = joblib.load(f'{MODELS_DIR}/svm.pkl')
    dt      = joblib.load(f'{MODELS_DIR}/dt.pkl')
    nb      = joblib.load(f'{MODELS_DIR}/nb.pkl')
    scaler  = joblib.load(f'{MODELS_DIR}/scaler.pkl')
    lt_enc  = joblib.load(f'{MODELS_DIR}/loan_type_enc.pkl')

    with open(f'{MODELS_DIR}/metrics.json') as f:
        METRICS = json.load(f)

    print("✅ All models loaded successfully")
    for name, m in METRICS.items():
        print(f"   {name}: {m['accuracy']}% test accuracy, {m['cv']}% CV")

except FileNotFoundError:
    print("❌ Models not found. Please run: python train_models.py first!")
    rf = svm = dt = nb = scaler = lt_enc = None
    METRICS = {}


def build_features(data, loan_type):
    """Convert raw form inputs into model feature vector."""
    income      = float(data.get('income', 0))
    cibil       = float(data.get('cibil', 750))
    loan_amount = float(data.get('loanAmount', 0))
    tenure      = float(data.get('tenure', 60))
    emi         = float(data.get('emi', 0))
    age         = float(data.get('age', 30))

    # Encode loan type (handle unseen labels gracefully)
    known = list(lt_enc.classes_)
    if loan_type not in known:
        loan_type = known[0]
    lt_encoded = lt_enc.transform([loan_type])[0]

    emi_ratio   = (emi / income * 100) if income > 0 else 0
    loan_to_inc = loan_amount / (income * 12) if income > 0 else 0

    return np.array([[age, income, cibil, loan_amount, tenure, emi, emi_ratio, loan_to_inc, lt_encoded]])


def predict_all(features):
    """Run all 4 models and return results."""
    features_sc = scaler.transform(features)

    results = {}
    for name, model, use_scaled in [
        ('Random Forest',  rf,  False),
        ('SVM',            svm, True),
        ('Decision Tree',  dt,  False),
        ('Naive Bayes',    nb,  True),
    ]:
        X = features_sc if use_scaled else features
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        conf  = round(float(max(proba)) * 100, 1)
        results[name] = {
            'approved':    bool(pred == 1),
            'confidence':  conf,
            'accuracy':    METRICS.get(name, {}).get('accuracy', 0)
        }

    return results


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': rf is not None, 'metrics': METRICS})


@app.route('/predict/home', methods=['POST'])
def predict_home():
    if rf is None:
        return jsonify({'error': 'Models not loaded. Run train_models.py first.'}), 503

    d = request.json
    # Estimate EMI from loan amount if not provided
    income      = float(d.get('income', 0))
    loan_amount = float(d.get('loanAmount', 0))
    tenure      = float(d.get('tenure', 240))
    emi         = float(d.get('existingEmi', 0))
    # Estimate new EMI using rough formula (9% rate)
    monthly_rate = 0.09 / 12
    if tenure > 0 and loan_amount > 0:
        estimated_emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)
        total_emi = emi + estimated_emi
    else:
        total_emi = emi

    data = {
        'income': income, 'cibil': d.get('cibil', 750),
        'loanAmount': loan_amount, 'tenure': tenure,
        'emi': total_emi, 'age': d.get('age', 30)
    }

    features = build_features(data, 'Home')
    predictions = predict_all(features)

    # Ensemble: majority vote
    votes = sum(1 for r in predictions.values() if r['approved'])
    ensemble_approved = votes >= 2
    ensemble_conf = round(np.mean([r['confidence'] for r in predictions.values()]), 1)

    # Key factors
    factors = []
    cibil = float(d.get('cibil', 750))
    emi_ratio = (total_emi / income * 100) if income > 0 else 0
    lti = loan_amount / (income * 12) if income > 0 else 0

    if cibil >= 750:  factors.append({'t': f'Excellent CIBIL Score ({int(cibil)})', 'p': True})
    elif cibil >= 700: factors.append({'t': f'Good CIBIL Score ({int(cibil)})', 'p': True})
    else:              factors.append({'t': f'Low CIBIL Score ({int(cibil)})', 'p': False})

    if emi_ratio <= 30: factors.append({'t': f'Healthy EMI ratio ({emi_ratio:.0f}%)', 'p': True})
    elif emi_ratio <= 50: factors.append({'t': f'Moderate EMI burden ({emi_ratio:.0f}%)', 'p': True})
    else:               factors.append({'t': f'High EMI burden ({emi_ratio:.0f}%)', 'p': False})

    if lti <= 36:  factors.append({'t': 'Healthy loan-to-income ratio', 'p': True})
    elif lti > 50: factors.append({'t': 'High loan relative to income', 'p': False})

    if income >= 100000: factors.append({'t': 'High income bracket', 'p': True})
    elif income < 25000: factors.append({'t': 'Income may be insufficient', 'p': False})

    emp = d.get('employment', '')
    if emp == 'govt':      factors.append({'t': 'Government employment', 'p': True})
    elif emp == 'salaried': factors.append({'t': 'Stable salaried employment', 'p': True})

    return jsonify({
        'approved': ensemble_approved,
        'confidence': ensemble_conf,
        'models': [
            {'name': name, 'approved': r['approved'], 'conf': r['confidence'], 'accuracy': r['accuracy']}
            for name, r in predictions.items()
        ],
        'factors': factors[:6],
        'source': 'real_ml'
    })


@app.route('/predict/education', methods=['POST'])
def predict_education():
    if rf is None:
        return jsonify({'error': 'Models not loaded. Run train_models.py first.'}), 503

    d = request.json
    income      = float(d.get('parentIncome', 0))
    loan_amount = float(d.get('loanAmount', 0))
    tenure      = float(d.get('tenure', 120))
    monthly_rate = 0.085 / 12
    if tenure > 0 and loan_amount > 0:
        emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure / ((1 + monthly_rate)**tenure - 1)
    else:
        emi = 0

    data = {
        'income': income, 'cibil': 720,   # education loans don't use student CIBIL
        'loanAmount': loan_amount, 'tenure': tenure,
        'emi': emi, 'age': d.get('age', 22)
    }

    features = build_features(data, 'Education')
    predictions = predict_all(features)

    votes = sum(1 for r in predictions.values() if r['approved'])
    ensemble_approved = votes >= 2
    ensemble_conf = round(np.mean([r['confidence'] for r in predictions.values()]), 1)

    factors = []
    inst = d.get('institution', '')
    score = float(d.get('academicScore', 75))
    collateral = d.get('collateral', 'none')
    lti = loan_amount / (income * 12) if income > 0 else 0

    if inst == 'iit_iim':   factors.append({'t': 'Premier institution (IIT/IIM/AIIMS)', 'p': True})
    elif inst == 'central': factors.append({'t': 'Reputed Central University', 'p': True})
    elif inst == 'deemed':  factors.append({'t': 'Deemed University', 'p': True})
    else:                   factors.append({'t': 'Private institution', 'p': False})

    if score >= 85:   factors.append({'t': f'Excellent academics ({score:.0f}%)', 'p': True})
    elif score >= 70: factors.append({'t': f'Good academic record ({score:.0f}%)', 'p': True})
    else:             factors.append({'t': f'Average academics ({score:.0f}%)', 'p': False})

    if collateral == 'property': factors.append({'t': 'Property collateral provided', 'p': True})
    elif collateral == 'fd':     factors.append({'t': 'FD/LIC collateral available', 'p': True})
    elif collateral == 'none':   factors.append({'t': 'No collateral — higher risk', 'p': False})

    if lti <= 24:  factors.append({'t': 'Feasible loan-to-income ratio', 'p': True})
    elif lti > 60: factors.append({'t': 'Very high loan relative to income', 'p': False})

    if income >= 80000: factors.append({'t': 'Good parent income', 'p': True})
    elif income < 20000: factors.append({'t': 'Limited parent income', 'p': False})

    return jsonify({
        'approved': ensemble_approved,
        'confidence': ensemble_conf,
        'models': [
            {'name': name, 'approved': r['approved'], 'conf': r['confidence'], 'accuracy': r['accuracy']}
            for name, r in predictions.items()
        ],
        'factors': factors[:6],
        'source': 'real_ml'
    })


if __name__ == '__main__':
    print("\n🚀 LoanIQ ML Server starting on http://localhost:5050")
    print("   Make sure you ran train_models.py first!\n")
    app.run(host='127.0.0.1', port=5050, debug=False)
