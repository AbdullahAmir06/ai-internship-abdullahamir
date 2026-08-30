const BASE = "";

export async function checkHealth() {
  const t0 = performance.now();
  const res = await fetch(`${BASE}/healthz`);
  const ms = Math.round(performance.now() - t0);
  if (!res.ok) throw new Error(`status ${res.status}`);
  const data = await res.json();
  return { ...data, ms };
}

export async function predict(text, model = "a") {
  const res = await fetch(`${BASE}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  const data = await res.json();
  if (!res.ok) {
    const err = new Error(data.detail || data.error || `request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

export async function fetchModels() {
  const res = await fetch(`${BASE}/api/v1/models`);
  if (!res.ok) throw new Error(`status ${res.status}`);
  const data = await res.json();
  return data.models;
}

export async function adversarialCheck(text) {
  const res = await fetch(`${BASE}/api/v1/adversarial-check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  if (!res.ok) {
    const err = new Error(data.detail || data.error || `request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}
