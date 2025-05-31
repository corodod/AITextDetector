from flask import Flask, render_template, request
from model_utils import load_all, predict_text

app = Flask(__name__)
tokenizer, encoder_model, mlp_model, device = load_all()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_text = request.form["text"]
        prob = predict_text(user_text, tokenizer, encoder_model, mlp_model, device)
        prediction = f"Вероятность генерации ИИ: {prob * 100:.2f}%"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
