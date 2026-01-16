let keystrokes = [];
let lastTime = null;
let analyzed = false;

const textarea = document.getElementById("typingArea");
const canvas = document.getElementById("typingChart");
const ctx = canvas.getContext("2d");

/**
 * Draws a flat baseline when the system is idle or cleared.
 */
function drawBaseline() {
  ctx.clearRect(0, 0, 260, 120);
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 60);
  ctx.lineTo(260, 60);
  ctx.stroke();
}

// Initial draw
drawBaseline();

/**
 * Captures the timing (latency) between keypresses.
 */
textarea.addEventListener("keydown", () => {
  if (analyzed) return;

  const now = performance.now();
  if (lastTime !== null) {
    keystrokes.push(now - lastTime);
  }
  lastTime = now;

  document.getElementById("charCount").innerText =
    textarea.value.length + " characters";
});

/**
 * Resets all data, UI labels, and the visualization graph.
 */
function clearInput() {
  textarea.value = "";
  keystrokes = [];
  lastTime = null;
  analyzed = false;
  document.getElementById("charCount").innerText = "0 characters";
  document.getElementById("result").innerText = "Idle";
  document.getElementById("confidenceBox").innerText = "0%";
  document.getElementById("donut").style.strokeDashoffset = 314;
  drawBaseline();
}

/**
 * Generates a real waveform based on collected keystroke timing data.
 * Faster typing results in higher peaks on the graph.
 */
function drawWaveform() {
  ctx.clearRect(0, 0, 260, 120);
  if (keystrokes.length < 2) {
    drawBaseline();
    return;
  }

  const min = Math.min(...keystrokes);
  const max = Math.max(...keystrokes);
  const range = max - min || 1;

  ctx.strokeStyle = "#F59E0B"; // Golden accent for the rhythm
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.beginPath();

  keystrokes.forEach((v, i) => {
    const x = (i / (keystrokes.length - 1)) * 260;
    const normalized = (v - min) / range;
    // Fast typing (lower latency) results in a higher Y-coordinate peak
    const y = 100 - normalized * 80;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });

  ctx.stroke();
}

/**
 * Sends keystroke and text data to the Flask backend for analysis.
 */
function sendData() {
  if (keystrokes.length < 10) {
    alert("Please type at least 10 characters before analyzing.");
    return;
  }

  // Prevent further data collection during current analysis
  analyzed = true;

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data: keystrokes,
      text: textarea.value
    })
  })
    .then(res => res.json())
    .then(result => {
      // Update Emotion and Method Status
      document.getElementById("result").innerText = result.emotion;
      document.getElementById("methodIndicator").innerText =
        result.emotion.includes("NLP") ? "Status: Context-Aware NLP Active" : "Status: AI Keystroke Active";

      // Process the mathematical confidence score from the backend equations
      const percent = result.confidence
        ? Math.round(result.confidence * 100)
        : 0;

      // Update UI Text elements
      document.getElementById("confidenceBox").innerText = percent + "%";
      const pill = document.getElementById("confidencePill");
      if (pill) pill.innerText = percent + "%";

      // Update the Donut Chart visual
      const circumference = 314;
      document.getElementById("donut").style.strokeDashoffset =
        circumference - (percent / 100) * circumference;

      // Draw the actual rhythm waveform captured during typing
      drawWaveform();
    })
    .catch(err => {
      console.error("Error:", err);
      analyzed = false; // Allow retry on error
    });
}