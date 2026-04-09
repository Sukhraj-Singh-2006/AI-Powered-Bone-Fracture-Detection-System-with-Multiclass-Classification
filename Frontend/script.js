// ===========================
// AUTO TRANSITION
// ===========================

window.onload = function () {
  setTimeout(function () {
    document.getElementById("welcomePage").style.display = "none";
    document.getElementById("mainApp").style.display = "block";
  }, 3000);
};

// ===========================
// TYPING ANIMATION
// ===========================

const text = "Bone Fracture Detection Model";
let index = 0;

function typeEffect() {
  if (index < text.length) {
    document.getElementById("typedText").innerHTML += text.charAt(index);
    index++;
    setTimeout(typeEffect, 80);
  }
}

typeEffect();

// ===========================
// NEURAL NETWORK BACKGROUND
// ===========================

const canvas = document.getElementById("networkCanvas");
const ctx = canvas.getContext("2d");

const networkColors = {
  particle: "#f97316",
  line: "rgba(168,85,247,0.18)"
};

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const particles = [];

for (let i = 0; i < 80; i++) {
  particles.push({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    vx: (Math.random() - 0.5) * 1,
    vy: (Math.random() - 0.5) * 1,
  });
}

function animate() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  particles.forEach((p) => {
    p.x += p.vx;
    p.y += p.vy;

    if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
    if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

    ctx.beginPath();
    ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = networkColors.particle;
    ctx.fill();

    particles.forEach((p2) => {
      const dist = Math.hypot(p.x - p2.x, p.y - p2.y);

      if (dist < 120) {
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = networkColors.line;
        ctx.stroke();
      }
    });
  });

  requestAnimationFrame(animate);
}

animate();

window.addEventListener("resize", function () {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
});

// ===========================
// IMAGE PREVIEW
// ===========================

const uploadInput = document.getElementById("imageUpload");
const previewImage = document.getElementById("previewImage");
const statusBadge = document.getElementById("statusBadge");
const gaugeFill = document.getElementById("gaugeFill");
const gaugeContainer = document.querySelector(".gauge-container");
const analyzeButton = document.querySelector(".upload-card button");

function resetStatusAppearance() {
  statusBadge.classList.remove("is-loading", "status-fracture", "status-clear");
  gaugeContainer.classList.remove("is-alert", "is-clear");
}

function setLoadingState(isLoading) {
  if (isLoading) {
    resetStatusAppearance();
    statusBadge.classList.add("is-loading");
    analyzeButton.classList.add("is-loading");
    analyzeButton.disabled = true;
  } else {
    statusBadge.classList.remove("is-loading");
    analyzeButton.classList.remove("is-loading");
    analyzeButton.disabled = false;
  }
}

function updateDiagnosisAppearance(result) {
  resetStatusAppearance();

  const normalizedResult = String(result).toLowerCase();

  if (normalizedResult.includes("no fracture")) {
    statusBadge.classList.add("status-clear");
    gaugeContainer.classList.add("is-clear");
  } else if (normalizedResult.includes("fracture")) {
    statusBadge.classList.add("status-fracture");
    gaugeContainer.classList.add("is-alert");
  }
}

uploadInput.addEventListener("change", function () {
  const file = this.files[0];

  if (file) {
    const imageURL = URL.createObjectURL(file);
    previewImage.src = imageURL;
    previewImage.style.display = "block";
  }
});

// ===========================
// ANALYZE IMAGE (API CALL)
// ===========================

function analyzeImage() {

  const file = uploadInput.files[0];

  if (!file) {
    alert("Please upload an X-ray image");
    return;
  }

  statusBadge.innerText = "AI Analyzing...";
  setLoadingState(true);

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://localhost:5000/predict", {
    method: "POST",
    body: formData
  })

  .then((response) => {
    if (!response.ok) {
      throw new Error("Server error");
    }
    return response.json();
  })

  .then((data) => {

    // =====================
    // DIAGNOSIS RESULT
    // =====================

    statusBadge.innerText = data.result;
    updateDiagnosisAppearance(data.result);

    document.getElementById("confidenceText").innerText =
      "Confidence: " + data.confidence + "%";

    document.getElementById("fractureType").innerText =
      "Fracture Type: " + data.type;

    document.getElementById("typeConfidence").innerText =
      "Type Confidence: " + data.type_confidence + "%";


    // =====================
    // REPORT SECTION
    // =====================

    document.getElementById("resultText").innerText =
      "Result: " + data.result;

    document.getElementById("confidenceReport").innerText =
      "Confidence: " + data.confidence + "%";

    document.getElementById("typeReport").innerText =
      "Fracture Type: " + data.type;

    document.getElementById("typeConfidenceReport").innerText =
      "Type Confidence: " + data.type_confidence + "%";


    document.getElementById("recommendation").innerText =
      data.recommendation ||
      "Please consult an orthopedic specialist for professional medical evaluation.";

    document.getElementById("note").innerText =
      data.note ||
      "This AI system provides assistance only and should not replace clinical diagnosis.";


    // =====================
    // UPDATE CONFIDENCE GAUGE
    // =====================

    const confidence = data.confidence;

    document.getElementById("gaugeText").innerText = confidence + "%";

    const circumference = 251;
    const offset = circumference - (confidence / 100) * circumference;

    gaugeFill.style.strokeDashoffset = offset;

    setLoadingState(false);

  })

  .catch((err) => {
    console.log(err);
    setLoadingState(false);
    statusBadge.innerText = "Analysis failed";
    alert("Image Format is not supported. Please use JPEG or PNG files.");
  });

}

// ===========================
// GOOGLE MAPS LOCATION SEARCH
// ===========================

// ===========================
// FIND NEARBY X-RAY CENTER
// ===========================

function findXrayCenter() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(function (position) {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;

      const url = `https://www.google.com/maps/search/x-ray+center/@${lat},${lon},15z`;

      window.open(url, "_blank");
    });
  } else {
    alert("Geolocation not supported by this browser.");
  }
}

// ===========================
// FIND NEARBY HOSPITAL
// ===========================

function findHospital() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(function (position) {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;

      const url = `https://www.google.com/maps/search/hospital/@${lat},${lon},15z`;

      window.open(url, "_blank");
    });
  } else {
    alert("Geolocation not supported by this browser.");
  }
}

// ===========================
// DOWNLOAD PDF REPORT
// ===========================
function downloadReport() {

const { jsPDF } = window.jspdf;
const doc = new jsPDF();

const result = document.getElementById("resultText").innerText;
const confidence = document.getElementById("confidenceReport").innerText;
const type = document.getElementById("typeReport").innerText;
const typeConfidence = document.getElementById("typeConfidenceReport").innerText;
const recommendation = document.getElementById("recommendation").innerText;
const note = document.getElementById("note").innerText;

const img = document.getElementById("previewImage");

if (!result) {
alert("Please analyze the X-ray first.");
return;
}

doc.setFontSize(18);
doc.text("Bone Fracture Detection AI Report", 20, 20);

const today = new Date().toLocaleDateString();

doc.setFontSize(12);
doc.text("Report Date: " + today, 20, 30);

// Add X-ray Image
if (img.src) {
doc.addImage(img.src, "JPEG", 20, 40, 70, 70);
}

// Diagnosis text
doc.text(result, 20, 120);
doc.text(confidence, 20, 130);
doc.text(type, 20, 140);
doc.text(typeConfidence, 20, 150);

doc.text("Recommendation:", 20, 170);
doc.text(recommendation, 20, 180);

doc.text("Note:", 20, 200);
doc.text(note, 20, 210);

doc.save("AI_Fracture_Report.pdf");
}
