
import torch
from PIL import Image
from torchvision import transforms
from model import create_model
import torch.nn.functional as F
from datetime import datetime
import numpy as np

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===============================
# LABELS
# ===============================
binary_classes = ["fractured", "not_fractured"]

type_classes = [
    "Avulsion","Comminuted","Compression","Fracture","Greenstick",
    "Hairline","Impacted","Intra-articular","Longitudinal",
    "Oblique","Pathological","Spiral"
]

# ===============================
# LOAD MODELS
# ===============================
binary_model = create_model(2)
binary_model.load_state_dict(
    torch.load("models/binary_model.pth", map_location=device)
)
binary_model.to(device)
binary_model.eval()

type_model = create_model(12)
type_model.load_state_dict(
    torch.load("models/type_model.pth", map_location=device)
)
type_model.to(device)
type_model.eval()

# ===============================
# 🔥 X-RAY VALIDATION FUNCTION
# ===============================
def is_xray_image(image):
    img = np.array(image)

    if len(img.shape) == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

        # Check color variation
        diff = np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b))

        if diff < 20:  # X-ray images are mostly grayscale
            return True

    return False

# ===============================
# REPORT GENERATION
# ===============================
def generate_report(image_name, result, confidence,
                    fracture_type=None, type_confidence=None):

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
====================================================
              AI RADIOLOGY DIAGNOSTIC REPORT
====================================================

Report Generated : {current_time}
Image File       : {image_name}

----------------------------------------------------
DIAGNOSIS SUMMARY
----------------------------------------------------
Finding         : {result}
Confidence      : {confidence:.2f}%
"""

    if fracture_type:
        report += f"""
----------------------------------------------------
FRACTURE CLASSIFICATION
----------------------------------------------------
Fracture Type   : {fracture_type}
Type Confidence : {type_confidence:.2f}%
"""
    else:
        report += """
----------------------------------------------------
FRACTURE CLASSIFICATION
----------------------------------------------------
Fracture Type   : N/A
Type Confidence : N/A
"""

    report += """
----------------------------------------------------
CLINICAL RECOMMENDATION
----------------------------------------------------
• Immediate consultation with an orthopedic specialist is advised.
• Further imaging (CT/MRI) may be required.
• Avoid stress on affected area.

----------------------------------------------------
DISCLAIMER
----------------------------------------------------
This AI system is for assistive purposes only and does
NOT replace professional medical diagnosis.

====================================================
"""

    with open("diagnosis_report.txt", "w") as f:
        f.write(report)

    return report

# ===============================
# MAIN PIPELINE (CLI)
# ===============================
def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    # 🔥 VALIDATION
    if not is_xray_image(image):
        print("\n❌ Invalid Image: Please upload a valid X-ray image\n")
        return

    image_tensor = transform(image).unsqueeze(0).to(device)

    # STEP 1
    with torch.no_grad():
        output = binary_model(image_tensor)
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)
        result = binary_classes[pred.item()]
        confidence_score = confidence.item() * 100

        if result == "not_fractured":
            report = generate_report(
                image_path,
                "No Fracture Detected",
                confidence_score
            )
            print(report)
            return

    # STEP 2
    with torch.no_grad():
        output = type_model(image_tensor)
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)
        fracture_type = type_classes[pred.item()]
        type_confidence = confidence.item() * 100

    report = generate_report(
        image_path,
        "Fracture Detected",
        confidence_score,
        fracture_type,
        type_confidence
    )

    print(report)

# ===============================
# API FUNCTION (FLASK)
# ===============================
def predict_image(image):

    # 🔥 VALIDATION
    if not is_xray_image(image):
        return {
            "result": "Invalid Image",
            "confidence": 0,
            "type": "Not an X-ray",
            "type_confidence": 0
        }

    image_tensor = transform(image).unsqueeze(0).to(device)

    # STEP 1
    with torch.no_grad():
        output = binary_model(image_tensor)
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)
        result = binary_classes[pred.item()]
        confidence_score = confidence.item() * 100

        if result == "not_fractured":
            return {
                "result": "No Fracture Detected",
                "confidence": round(confidence_score, 2),
                "type": "None",
                "type_confidence": 0
            }

    # STEP 2
    with torch.no_grad():
        output = type_model(image_tensor)
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)
        fracture_type = type_classes[pred.item()]
        type_confidence = confidence.item() * 100

    return {
        "result": "Fracture Detected",
        "confidence": round(confidence_score, 2),
        "type": fracture_type,
        "type_confidence": round(type_confidence, 2)
    }

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    path = input("Enter image path: ")
    predict(path)

