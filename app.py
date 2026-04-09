from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from predict_pipeline import predict_image

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file).convert("RGB")

        result = predict_image(image)
        print("RESULT:", result)

        return jsonify({
            "result": result["result"],
            "confidence": float(result["confidence"]),
            "type": result["type"],
            "type_confidence": float(result["type_confidence"])
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)