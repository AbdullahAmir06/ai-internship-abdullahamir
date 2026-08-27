// Part C -- interactive client. Relative URLs throughout, so the identical
// static files work unmodified locally, in Docker, or on the live deploy.

const $ = (sel) => document.querySelector(sel);

async function checkHealth() {
  const badge = $("#health-badge");
  try {
    const t0 = performance.now();
    const res = await fetch("/healthz");
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();
    badge.textContent = `API online · ${ms}ms · model ${data.model_loaded ? "loaded" : "not loaded"}`;
    badge.className = "badge badge-ok";
  } catch (err) {
    badge.textContent = `API unreachable (${err.message})`;
    badge.className = "badge badge-err";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

$("#analyze-btn").addEventListener("click", async () => {
  const btn = $("#analyze-btn");
  const statusEl = $("#predict-status");
  const resultEl = $("#predict-result");
  const text = $("#review-text").value;

  if (!text.trim()) {
    statusEl.textContent = "Please enter some review text first.";
    statusEl.className = "status err";
    return;
  }

  btn.disabled = true;
  resultEl.className = "result";
  statusEl.textContent = "Analyzing…";
  statusEl.className = "status loading";
  const t0 = performance.now();

  try {
    const res = await fetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    const ms = Math.round(performance.now() - t0);

    if (!res.ok) {
      statusEl.textContent = `Error ${res.status}: ${data.detail || data.error || "request failed"}`;
      statusEl.className = "status err";
      return;
    }

    statusEl.textContent = `Done in ${ms}ms`;
    statusEl.className = "status ok";
    resultEl.className = `result visible ${data.label}`;
    resultEl.innerHTML = `<span class="label">${data.label}</span>` +
      `<span class="confidence">confidence: ${(data.confidence * 100).toFixed(1)}%</span>`;
  } catch (err) {
    statusEl.textContent = `Network error: ${err.message}`;
    statusEl.className = "status err";
  } finally {
    btn.disabled = false;
  }
});

async function loadComparison() {
  const container = $("#comparison-table-container");
  try {
    const res = await fetch("/api/v1/models");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();

    let html = `<table class="comparison"><thead><tr>
      <th>Model</th><th>Approach</th><th>Test Acc</th><th>Test F1</th>
      <th>Latency</th><th>Size</th><th>Deployed</th>
    </tr></thead><tbody>`;
    for (const m of data.models) {
      html += `<tr>
        <td>${m.name}</td>
        <td>${m.approach}</td>
        <td>${m.test_accuracy != null ? (m.test_accuracy * 100).toFixed(1) + "%" : "—"}</td>
        <td>${m.test_macro_f1 != null ? m.test_macro_f1.toFixed(3) : "—"}</td>
        <td>${m.avg_latency_ms != null ? m.avg_latency_ms.toFixed(2) + "ms" : "—"}</td>
        <td>${m.artifact_size || "—"}</td>
        <td class="${m.deployed ? "deployed-yes" : "deployed-no"}">${m.deployed ? "Yes" : "No"}</td>
      </tr>`;
    }
    html += "</tbody></table>";
    for (const m of data.models) {
      if (m.note) html += `<div class="model-note"><strong>${m.name}:</strong> ${m.note}</div>`;
    }
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<div class="status err">Could not load comparison data: ${err.message}</div>`;
  }
}
loadComparison();
