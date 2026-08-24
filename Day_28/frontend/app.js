// Part D.1 -- interactive client, communicating asynchronously with the API.
// Uses relative URLs ("/api/v1/...", "/healthz") throughout rather than a
// hard-coded host, so the exact same static files work unmodified whether
// loaded from http://localhost:8000 (local dev), a docker-compose instance,
// or the live Hugging Face Spaces URL -- the frontend always talks to
// whatever origin served it.

const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#panel-${btn.dataset.tab}`).classList.add("active");
  });
});

// ---------------------------------------------------------------- health badge
async function checkHealth() {
  const badge = $("#health-badge");
  try {
    const t0 = performance.now();
    const res = await fetch("/healthz");
    const ms = Math.round(performance.now() - t0);
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();
    badge.textContent = `API online · ${ms}ms · ${data.loaded_models.length} model(s) loaded`;
    badge.className = "badge badge-ok";
  } catch (err) {
    badge.textContent = `API unreachable (${err.message})`;
    badge.className = "badge badge-err";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

// ---------------------------------------------------------------- generation param live labels
const tempSlider = $("#generate-temperature");
const topPSlider = $("#generate-top-p");
tempSlider.addEventListener("input", () => { $("#generate-temperature-val").textContent = tempSlider.value; });
topPSlider.addEventListener("input", () => { $("#generate-top-p-val").textContent = topPSlider.value; });

const strategySelect = $("#generate-strategy");
function updateSamplingParamsVisibility() {
  const strat = strategySelect.value;
  const showTemp = strat === "top_k" || strat === "top_p" || strat === "temperature";
  const showTopP = strat === "top_p";
  const showTopK = strat === "top_k";
  const showBeams = strat === "beam";
  $("#generate-temperature").closest("label").style.display = showTemp ? "" : "none";
  $("#generate-top-p").closest("label").style.display = showTopP ? "" : "none";
  $("#generate-top-k").closest("label").style.display = showTopK ? "" : "none";
  $("#generate-num-beams").closest("label").style.display = showBeams ? "" : "none";
}
strategySelect.addEventListener("change", updateSamplingParamsVisibility);
updateSamplingParamsVisibility();

// ---------------------------------------------------------------- request builders
function buildPayload(panel) {
  if (panel === "sentiment") {
    return { text: $("#sentiment-text").value };
  }
  if (panel === "summarize") {
    return {
      text: $("#summarize-text").value,
      min_length: parseInt($("#summarize-min").value, 10),
      max_length: parseInt($("#summarize-max").value, 10),
    };
  }
  if (panel === "generate") {
    return {
      prompt: $("#generate-text").value,
      max_new_tokens: parseInt($("#generate-max-tokens").value, 10),
      decoding_strategy: strategySelect.value,
      temperature: parseFloat(tempSlider.value),
      top_p: parseFloat(topPSlider.value),
      top_k: parseInt($("#generate-top-k").value, 10),
      num_beams: parseInt($("#generate-num-beams").value, 10),
    };
  }
}

function renderResult(panel, data) {
  const box = $(`#${panel}-result`);
  let body = "";
  if (panel === "sentiment") {
    body = `Label: ${data.label}\nConfidence: ${(data.score * 100).toFixed(2)}%`;
  } else if (panel === "summarize") {
    body = data.summary;
  } else if (panel === "generate") {
    body = data.generated_text;
  }
  const meta = data.meta
    ? `\n\n---\nlatency: ${data.latency_ms.toFixed(1)}ms · tokens: ${data.meta.token_count}` +
      (data.meta.truncated ? " (input truncated)" : "")
    : "";
  box.textContent = body + meta;
  box.classList.add("visible");
}

// ---------------------------------------------------------------- run buttons
document.querySelectorAll(".run-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const panel = btn.dataset.panel;
    const endpoint = btn.dataset.endpoint;
    const statusEl = $(`#${panel}-status`);
    const resultEl = $(`#${panel}-result`);
    const payload = buildPayload(panel);

    btn.disabled = true;
    resultEl.classList.remove("visible");
    statusEl.textContent = "Running inference…";
    statusEl.className = "status loading";
    const t0 = performance.now();

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      const ms = Math.round(performance.now() - t0);

      if (!res.ok) {
        statusEl.textContent = `Error ${res.status}: ${data.detail || data.error || "request failed"}`;
        statusEl.className = "status err";
        resultEl.classList.remove("visible");
        return;
      }

      statusEl.textContent = `Done in ${ms}ms (round trip)`;
      statusEl.className = "status ok";
      renderResult(panel, data);
    } catch (err) {
      statusEl.textContent = `Network error: ${err.message}`;
      statusEl.className = "status err";
    } finally {
      btn.disabled = false;
    }
  });
});
