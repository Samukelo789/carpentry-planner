import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(BASE_DIR, filename)

def load_data():
    df = pd.read_csv(_path("Maker-Dataset.csv"))
    return df

def train_models():
    print("Loading maker dataset...")
    df = load_data()
    print(f"Loaded {len(df)} records")

    encoders = {}
    for col in ['product_idea', 'category', 'skill_level',
                'available_materials', 'available_tools', 'build_feasible']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # FIX: budget_zar added as a cost model feature.
    # budget_zar has 0.99 correlation with estimated_cost_zar.
    # Excluding it caused the same cost to be predicted for every
    # (category, skill_level) pair regardless of the user's budget.
    cost_features  = ['skill_level_enc', 'category_enc', 'budget_zar']
    other_features = ['skill_level_enc', 'category_enc']

    # MODEL 1: Cost Predictor
    print("\nTraining Model 1: Cost Predictor...")
    X_cost = df[cost_features]
    y_cost = df['estimated_cost_zar']
    Xtr, Xte, ytr, yte = train_test_split(X_cost, y_cost, test_size=0.2, random_state=42)
    cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
    cost_model.fit(Xtr, ytr)
    preds = cost_model.predict(Xte)
    mae = mean_absolute_error(yte, preds)
    r2  = r2_score(yte, preds)
    print(f"Cost Predictor - MAE: R{mae:.2f}  R2: {r2:.4f}")

    # MODEL 2: Feasibility Predictor
    print("\nTraining Model 2: Feasibility Predictor...")
    X_feas = df[other_features]
    y_feas = df['build_feasible_enc']
    Xtr, Xte, ytr, yte = train_test_split(X_feas, y_feas, test_size=0.2, random_state=42)
    feas_model = RandomForestClassifier(n_estimators=100, random_state=42)
    feas_model.fit(Xtr, ytr)
    acc = accuracy_score(yte, feas_model.predict(Xte))
    print(f"Feasibility Predictor - Accuracy: {acc*100:.1f}%")

    # MODEL 3: Complexity Classifier
    print("\nTraining Model 3: Complexity Classifier...")
    X_skill = df[other_features]
    y_skill = df['skill_level_enc']
    Xtr, Xte, ytr, yte = train_test_split(X_skill, y_skill, test_size=0.2, random_state=42)
    skill_model = RandomForestClassifier(n_estimators=100, random_state=42)
    skill_model.fit(Xtr, ytr)
    acc2 = accuracy_score(yte, skill_model.predict(Xte))
    print(f"Complexity Classifier - Accuracy: {acc2*100:.1f}%")

    print("\nSaving models...")
    with open(_path("cost_model.pkl"),  "wb") as f: pickle.dump(cost_model,  f)
    with open(_path("feas_model.pkl"),  "wb") as f: pickle.dump(feas_model,  f)
    with open(_path("skill_model.pkl"), "wb") as f: pickle.dump(skill_model, f)
    with open(_path("encoders.pkl"),    "wb") as f: pickle.dump(encoders,    f)
    print("All models saved!")


def load_models():
    with open(_path("cost_model.pkl"),  "rb") as f: cost_model  = pickle.load(f)
    with open(_path("feas_model.pkl"),  "rb") as f: feas_model  = pickle.load(f)
    with open(_path("skill_model.pkl"), "rb") as f: skill_model = pickle.load(f)
    with open(_path("encoders.pkl"),    "rb") as f: encoders    = pickle.load(f)
    return cost_model, feas_model, skill_model, encoders


def predict(category, skill_level, budget):
    """
    Predict cost, feasibility, and required skill.

    Parameters
    ----------
    category    : str  e.g. 'seating', 'tables', 'storage', 'outdoor', 'decorative'
    skill_level : str  e.g. 'Beginner', 'Intermediate', 'Advanced'
    budget      : int  user budget in ZAR
    """
    cost_model, feas_model, skill_model, encoders = load_models()

    try:
        category_enc = encoders['category'].transform([category.lower()])[0]
    except ValueError:
        category_enc = 0

    try:
        skill_enc = encoders['skill_level'].transform([skill_level])[0]
    except ValueError:
        skill_enc = 0

    try:
        budget_val = int(budget)
    except (TypeError, ValueError):
        budget_val = 100

    cost_input  = pd.DataFrame([[skill_enc, category_enc, budget_val]],
                               columns=['skill_level_enc', 'category_enc', 'budget_zar'])
    other_input = pd.DataFrame([[skill_enc, category_enc]],
                               columns=['skill_level_enc', 'category_enc'])

    raw_cost = round(cost_model.predict(cost_input)[0])
    raw_cost = max(1, min(raw_cost, budget_val))   # clamp to [1, budget]

    feas_enc    = feas_model.predict(other_input)[0]
    feasibility = encoders['build_feasible'].inverse_transform([feas_enc])[0]
    confidence  = round(max(feas_model.predict_proba(other_input)[0]) * 100, 1)

    skill_enc_pred = skill_model.predict(other_input)[0]
    req_skill   = encoders['skill_level'].inverse_transform([skill_enc_pred])[0]

    return {
        "predicted_cost": f"R{raw_cost}",
        "build_feasible": feasibility,
        "required_skill": req_skill,
        "confidence":     f"{confidence}%"
    }


if __name__ == "__main__":
    train_models()
    print("\n--- Sample predictions ---")
    for cat, skill, bud in [("seating","Beginner",300),("tables","Intermediate",1200),
                             ("storage","Advanced",2500),("outdoor","Beginner",180),("decorative","Intermediate",600)]:
        r = predict(cat, skill, bud)
        print(f"  {cat}/{skill}/R{bud} -> cost={r['predicted_cost']} feas={r['build_feasible']} conf={r['confidence']}")
