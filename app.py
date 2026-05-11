from flask import Flask, render_template, request, redirect, url_for
from groq import Groq
from database import get_db, init_db, get_similar_builds
from prompts import build_maker_prompt
from ml_model import predict
import os

# Manually load .env — no python-dotenv needed
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

app = Flask(__name__)

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Make sure .env exists with: GROQ_API_KEY=your_key_here")
client = Groq(api_key=api_key)

# -------------------------------------------------------
# ROUTE 1: Home
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------
# ROUTE 2: Submit — Run ML + LLM, save plan
# -------------------------------------------------------
@app.route("/submit", methods=["POST"])
def submit():
    maker_name   = request.form["maker_name"]
    product_idea = request.form["product_idea"]
    materials    = request.form["materials"]
    tools        = request.form["tools"]
    budget       = request.form["budget"]
    skill_level  = request.form["skill_level"]

    category_map = {
        # seating
        "chair":          "seating",
        "stool":          "seating",
        "bench":          "seating",
        "rocker":         "seating",
        "rocking":        "seating",
        "ottoman":        "seating",
        "footstool":      "seating",
        "chaise":         "seating",
        "lounge chair":   "seating",
        "porch swing":    "seating",
        "adirondack":     "seating",
        "pallet":         "seating",
        "high chair":     "seating",
        # tables
        "table":          "tables",
        "desk":           "tables",
        "workbench":      "tables",
        "countertop":     "tables",
        "island":         "tables",
        "butcher block":  "tables",
        # bedroom
        "bed":            "bedroom",
        "headboard":      "bedroom",
        "footboard":      "bedroom",
        "crib":           "bedroom",
        "nightstand":     "bedroom",
        "changing table": "bedroom",
        "armoire":        "bedroom",
        "wardrobe":       "bedroom",
        "dresser":        "bedroom",
        # storage
        "shelf":          "storage",
        "shelves":        "storage",
        "bookshelf":      "storage",
        "bookcase":       "storage",
        "rack":           "storage",
        "cabinet":        "storage",
        "box":            "storage",
        "crate":          "storage",
        "chest":          "storage",
        "credenza":       "storage",
        "sideboard":      "storage",
        "buffet":         "storage",
        "linen":          "storage",
        "shoe":           "storage",
        "coat":           "storage",
        "hall tree":      "storage",
        "tool chest":     "storage",
        "toolbox":        "storage",
        "spice":          "storage",
        "knife block":    "storage",
        "utensil":        "storage",
        "bread box":      "storage",
        "wine rack":      "storage",
        "jewelry":        "storage",
        "keepsake":       "storage",
        "hope chest":     "storage",
        "memory box":     "storage",
        "toy box":        "storage",
        "window seat":    "storage",
        # outdoor
        "planter":        "outdoor",
        "garden":         "outdoor",
        "trellis":        "outdoor",
        "arbor":          "outdoor",
        "pergola":        "outdoor",
        "gazebo":         "outdoor",
        "birdhouse":      "outdoor",
        "bird feeder":    "outdoor",
        "dog house":      "outdoor",
        "doghouse":       "outdoor",
        "cat tree":       "outdoor",
        "rabbit":         "outdoor",
        "chicken coop":   "outdoor",
        "pet ramp":       "outdoor",
        "step ladder":    "outdoor",
        "compost":        "outdoor",
        "gate":           "outdoor",
        # decorative
        "frame":          "decorative",
        "sign":           "decorative",
        "wall art":       "decorative",
        "carving":        "decorative",
        "cutting board":  "decorative",
        "serving tray":   "decorative",
        "clock":          "decorative",
        "mirror":         "decorative",
        # kids
        "dollhouse":      "kids",
        "toy":            "kids",
        "puzzle":         "kids",
        # workshop
        "sawhorse":       "workshop",
        "vise":           "workshop",
    }
    detected_category = "storage"
    product_lower = product_idea.lower()
    for keyword, cat in category_map.items():
        if keyword in product_lower:
            detected_category = cat
            break

    ml_predictions = predict(detected_category, skill_level, budget)

    # --- Tools-gap detection ---
    # If the maker listed no tools (or explicitly said "none"), flag it.
    tools_stripped = tools.strip().lower()
    if tools_stripped in ("", "none", "n/a", "no tools", "nothing"):
        tools_gap = "YES"
        ml_predictions["build_feasible"] = "TOOLS_MISSING"
        ml_predictions["confidence"]     = "N/A"
    else:
        tools_gap = "NO"

    history = get_similar_builds(
        product_idea=product_idea,
        skill_level=skill_level
    )

    prompt = build_maker_prompt(
        maker_name, product_idea, materials,
        tools, budget, skill_level,
        history, ml_predictions
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    llm_plan = response.choices[0].message.content

    conn = get_db()
    conn.execute("""
        INSERT INTO plans
        (maker_name, product_idea, materials, tools, budget,
         skill_level, llm_plan, ml_feasible, ml_cost,
         ml_skill, ml_confidence, ml_tools_gap)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (maker_name, product_idea, materials, tools, budget,
          skill_level, llm_plan,
          ml_predictions["build_feasible"],
          ml_predictions["predicted_cost"],
          ml_predictions["required_skill"],
          ml_predictions["confidence"],
          tools_gap))
    conn.commit()
    plan_id = conn.execute(
        "SELECT last_insert_rowid()"
    ).fetchone()[0]
    conn.close()

    return redirect(url_for("result", plan_id=plan_id))


# -------------------------------------------------------
# ROUTE 3: Result
# -------------------------------------------------------
@app.route("/result/<int:plan_id>")
def result(plan_id):
    conn = get_db()
    plan = conn.execute(
        "SELECT * FROM plans WHERE id = ?", (plan_id,)
    ).fetchone()
    conn.close()
    return render_template("result.html", plan=plan)


# -------------------------------------------------------
# ROUTE 4: Dashboard
# -------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    conn = get_db()
    plans = conn.execute(
        "SELECT * FROM plans ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", plans=plans)


# -------------------------------------------------------
# ROUTE 5: Update plan status
# -------------------------------------------------------
@app.route("/update/<int:plan_id>", methods=["POST"])
def update(plan_id):
    new_status = request.form["status"]
    assigned   = request.form["assigned_to"]
    conn = get_db()
    conn.execute("""
        UPDATE plans SET status = ?, assigned_to = ?
        WHERE id = ?
    """, (new_status, assigned, plan_id))
    conn.commit()
    conn.close()
    return redirect(url_for("dashboard"))


# -------------------------------------------------------
# ROUTE 6: Feedback loop — LLM revises plan
# -------------------------------------------------------
@app.route("/feedback/<int:plan_id>", methods=["POST"])
def feedback(plan_id):
    maker_feedback = request.form["feedback"]
    conn = get_db()
    plan = conn.execute(
        "SELECT * FROM plans WHERE id = ?", (plan_id,)
    ).fetchone()

    revision_prompt = f"""
You previously generated this carpentry prototype plan:

{plan['llm_plan']}

A local carpenter or workshop owner reviewed it and flagged these issues:
"{maker_feedback}"

Please revise the prototype plan addressing these concerns.
Keep the same format but adjust recommendations to be more
realistic for local conditions in South Africa.
All materials and tools suggested must be locally available.
All costs must be in South African Rands (ZAR).
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": revision_prompt}]
    )
    revised_plan = response.choices[0].message.content

    conn.execute(
        "UPDATE plans SET llm_plan = ? WHERE id = ?",
        (revised_plan, plan_id)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("result", plan_id=plan_id))


# -------------------------------------------------------
# ROUTE 7: Rate plan usefulness
# -------------------------------------------------------
@app.route("/rate/<int:plan_id>/<rating>")
def rate(plan_id, rating):
    if rating not in ["yes", "no"]:
        return redirect(url_for("result", plan_id=plan_id))
    conn = get_db()
    conn.execute(
        "UPDATE plans SET plan_useful = ? WHERE id = ?",
        ("YES" if rating == "yes" else "NO", plan_id)
    )
    conn.commit()
    conn.close()
    return redirect(url_for("result", plan_id=plan_id))


# -------------------------------------------------------
# ROUTE 8: Analytics
# -------------------------------------------------------
@app.route("/analytics")
def analytics():
    conn = get_db()
    plans = conn.execute(
        "SELECT * FROM plans ORDER BY created_at DESC"
    ).fetchall()
    conn.close()

    total    = len(plans)
    feasible = sum(1 for p in plans if p['ml_feasible'] == 'YES')
    done     = sum(1 for p in plans if p['status'] == 'Done')
    useful       = sum(1 for p in plans if p['plan_useful'] == 'YES') if plans and 'plan_useful' in plans[0].keys() else 0
    not_useful   = sum(1 for p in plans if p['plan_useful'] == 'NO')  if plans and 'plan_useful' in plans[0].keys() else 0
    rated        = useful + not_useful
    usability_rate = round((useful / rated * 100), 1) if rated > 0 else 0

    skill_data = {}
    for p in plans:
        s = p['skill_level']
        skill_data[s] = skill_data.get(s, 0) + 1

    product_data = {}
    for p in plans:
        prod = p['product_idea']
        product_data[prod] = product_data.get(prod, 0) + 1

    status_data = {}
    for p in plans:
        st = p['status']
        status_data[st] = status_data.get(st, 0) + 1

    budgets  = []
    ml_costs = []
    for p in plans:
        try:
            budgets.append(int(float(p['budget'])))
        except:
            budgets.append(0)
        try:
            ml_costs.append(int(float(p['ml_cost'].replace('R', ''))))
        except:
            ml_costs.append(0)

    plan_ids = [p['id'] for p in plans]
    avg_budget = f"R{int(sum(budgets)/len(budgets))}" if budgets else "R0"

    stats = {
        "total_plans":      total,
        "feasible_count":   feasible,
        "infeasible_count": total - feasible,
        "feasibility_rate": round((feasible/total*100), 1) if total > 0 else 0,
        "done_count":       done,
        "avg_budget":       avg_budget,
        "useful_count":     useful,
        "not_useful_count": not_useful,
        "usability_rate":   usability_rate,
        "rated_count":      rated,
        "feas_data":        {"yes": feasible, "no": total - feasible},
        "useful_data":      {"yes": useful, "no": not_useful},
        "skill_data":       skill_data,
        "product_data":     product_data,
        "status_data":      status_data,
        "budgets":          budgets,
        "ml_costs":         ml_costs,
        "plan_ids":         plan_ids,
    }

    return render_template("analytics.html", stats=stats, plans=plans)


# -------------------------------------------------------
# ROUTE 9: Ethics
# -------------------------------------------------------
@app.route("/ethics")
def ethics():
    return render_template("ethics.html")


# -------------------------------------------------------
# START
# -------------------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
