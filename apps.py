from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from quantums_model import HybridQCNN

app = Flask(__name__)
model = HybridQCNN()
model.load_state_dict(torch.load("hybrid_qcnn_model.pth"))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        try:
            sugar = float(request.form["blood_sugar"])
            pressure = float(request.form["blood_pressure"])
            vision = float(request.form["vision_score"])
            age = float(request.form["age"])
            input_data = np.array([[sugar, pressure, vision, age]])

            # Normalize like training
            scaler = pd.read_csv("train_quantum_ready.csv")
            mins = scaler.iloc[:, :-1].min().values
            maxs = scaler.iloc[:, :-1].max().values
            norm_input = (input_data - mins) / (maxs - mins)
            x_tensor = torch.tensor(norm_input).float()

            pred = torch.argmax(model(x_tensor), axis=1).item()
            prediction = "🩺 Diabetic Retinopathy Detected" if pred == 1 else "✅ No Diabetic Retinopathy"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("indexs.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
