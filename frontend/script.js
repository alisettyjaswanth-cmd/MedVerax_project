// MedVerax AI - Frontend Application Logic

// Auto-detect API Base URL
const API_BASE = (window.location.protocol.startsWith("http")) 
    ? window.location.origin 
    : "http://127.0.0.1:8000";

// DOM Elements
const claimInput = document.getElementById("claim-input");
const analyzeBtn = document.getElementById("analyze-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const apiDot = document.getElementById("api-status-dot");
const apiText = document.getElementById("api-status-text");

const resultCard = document.getElementById("result-card");
const placeholderState = document.getElementById("placeholder-state");
const activeResults = document.getElementById("active-results");
const riskBadge = document.getElementById("risk-badge");
const predictionVal = document.getElementById("prediction-val");
const confidenceVal = document.getElementById("confidence-val");
const confidenceBar = document.getElementById("confidence-bar");
const patternsBlock = document.getElementById("patterns-block");
const patternsList = document.getElementById("patterns-list");
const explanationText = document.getElementById("explanation-text");
const recommendationText = document.getElementById("recommendation-text");
const footerDisclaimer = document.getElementById("footer-disclaimer");
const historyTableBody = document.getElementById("history-table-body");

// Quick Example Helper
function setExample(text) {
    claimInput.value = text;
    claimInput.focus();
}

// Health Check API
async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        if (res.ok) {
            const data = await res.json();
            apiDot.className = "status-indicator status-online";
            apiText.innerText = "API Online";
        } else {
            throw new Error("Unhealthy response");
        }
    } catch (err) {
        apiDot.className = "status-indicator status-offline";
        apiText.innerText = "API Offline (Start FastAPI backend)";
    }
}

// Main Claim Analysis Handler
async function analyzeClaim() {
    const text = claimInput.value.trim();
    if (!text) {
        alert("Please enter a health claim or medical statement to analyze.");
        claimInput.focus();
        return;
    }

    // Set Loading State
    analyzeBtn.disabled = true;
    btnText.innerText = "Analyzing...";
    btnSpinner.classList.remove("hidden");

    try {
        const response = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server analysis error");
        }

        const data = await response.json();
        renderAnalysisResults(data);
        fetchHistory(); // Refresh history table

    } catch (error) {
        console.error("Analysis Error:", error);
        alert(`Analysis failed: ${error.message}. Make sure the FastAPI backend is running on ${API_BASE}.`);
    } finally {
        // Reset Loading State
        analyzeBtn.disabled = false;
        btnText.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Claim';
        btnSpinner.classList.add("hidden");
    }
}

// Render Results to UI
function renderAnalysisResults(data) {
    placeholderState.classList.add("hidden");
    activeResults.classList.remove("hidden");

    // 1. Risk Badge
    const risk = data.risk_level;
    riskBadge.innerText = risk;
    riskBadge.className = "risk-badge";

    if (risk === "High Risk") {
        riskBadge.classList.add("badge-high");
        confidenceBar.style.backgroundColor = "var(--risk-high)";
    } else if (risk === "Medium Risk") {
        riskBadge.classList.add("badge-medium");
        confidenceBar.style.backgroundColor = "var(--risk-med)";
    } else {
        riskBadge.classList.add("badge-low");
        confidenceBar.style.backgroundColor = "var(--risk-low)";
    }

    // 2. Metrics
    predictionVal.innerText = data.prediction;
    const confPercent = Math.round(data.confidence * 100);
    confidenceVal.innerText = `${confPercent}%`;
    confidenceBar.style.width = `${confPercent}%`;

    // 3. Detected Suspicious Patterns
    patternsList.innerHTML = "";
    if (data.detected_patterns && data.detected_patterns.length > 0) {
        patternsBlock.classList.remove("hidden");
        data.detected_patterns.forEach(item => {
            const chip = document.createElement("div");
            chip.className = "pattern-chip";
            chip.innerHTML = `
                <div class="pattern-name"><i class="fa-solid fa-triangle-exclamation"></i> ${item.category}: ${item.matched_phrases.join(", ")}</div>
                <div style="font-size: 12px; margin-top: 2px;">${item.explanation}</div>
            `;
            patternsList.appendChild(chip);
        });
    } else {
        patternsBlock.classList.add("hidden");
    }

    // 4. Explanation & Recommendation
    explanationText.innerText = data.explanation;
    recommendationText.innerText = data.safety_recommendation;

    // 5. Disclaimer
    if (data.disclaimer) {
        footerDisclaimer.innerText = data.disclaimer;
    }
}

// Fetch Previous Analyses from SQLite
async function fetchHistory() {
    try {
        const response = await fetch(`${API_BASE}/history?limit=10`);
        if (!response.ok) return;

        const data = await response.json();
        const items = data.history || [];

        if (items.length === 0) {
            historyTableBody.innerHTML = `
                <tr><td colspan="5" class="text-center">No previous analyses recorded yet.</td></tr>
            `;
            return;
        }

        historyTableBody.innerHTML = items.map(item => {
            let badgeClass = "badge-pending";
            if (item.risk_level === "High Risk") badgeClass = "badge-high";
            else if (item.risk_level === "Medium Risk") badgeClass = "badge-medium";
            else if (item.risk_level === "Low Risk") badgeClass = "badge-low";

            const shortClaim = item.claim.length > 60 
                ? item.claim.substring(0, 60) + "..." 
                : item.claim;

            return `
                <tr>
                    <td style="white-space: nowrap; color: var(--muted); font-size: 12px;">${item.created_at || "Just now"}</td>
                    <td><strong>${escapeHtml(shortClaim)}</strong></td>
                    <td><span class="risk-badge ${badgeClass}" style="font-size: 11px; padding: 3px 8px;">${item.risk_level}</span></td>
                    <td>${Math.round(item.confidence * 100)}%</td>
                    <td style="color: var(--muted); font-size: 13px;">${escapeHtml(item.prediction)}</td>
                </tr>
            `;
        }).join("");

    } catch (err) {
        console.warn("Could not fetch history:", err);
    }
}

// Clear History
async function clearAllHistory() {
    if (!confirm("Are you sure you want to clear all analysis history?")) return;
    try {
        await fetch(`${API_BASE}/history`, { method: "DELETE" });
        fetchHistory();
    } catch (err) {
        alert("Could not clear history.");
    }
}

// Helper: Escape HTML
function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    checkApiHealth();
    fetchHistory();
    setInterval(checkApiHealth, 10000); // Poll API health every 10s
});
