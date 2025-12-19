from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and threshold
model = joblib.load("best_lead_model.pkl")
best_th = joblib.load("best_threshold.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        # Get form inputs
        source = request.form.get("source")
        agent = request.form.get("agent")
        location = request.form.get("location")
        delivery_mode = request.form.get("delivery_mode")
        product = int(request.form.get("product"))
        dow = int(request.form.get("dow"))
        month = int(request.form.get("month"))
        quarter = int(request.form.get("quarter"))
        year = int(request.form.get("year"))

        # Build input DataFrame
        df = pd.DataFrame([{
            "Source_Cleaned": source,
            "Sales_Agent": agent,
            "Location": location,
            "Product_ID": product,
            "day_of_week": dow,
            "month_num": month,
            "quarter": quarter,
            "year": year,
            "Delivery_Mode_Mode-2": 1 if delivery_mode=="Mode-2" else 0,
            "Delivery_Mode_Mode-3": 1 if delivery_mode=="Mode-3" else 0,
            "Delivery_Mode_Mode-4": 1 if delivery_mode=="Mode-4" else 0,
            "Delivery_Mode_Mode-5": 1 if delivery_mode=="Mode-5" else 0,
        }])

        # Predict
        prob = model.predict_proba(df)[0][1] * 100
        prediction = "HIGH POTENTIAL" if prob >= 50 else "LOW POTENTIAL"
        confidence = round(prob, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence, threshold=best_th)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


