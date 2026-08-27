import { useEffect, useState } from "react";
import { checkHealth } from "../api";

export default function HealthBadge() {
  const [state, setState] = useState({ status: "checking" });

  useEffect(() => {
    let alive = true;
    const run = async () => {
      try {
        const data = await checkHealth();
        if (alive) setState({ status: "ok", ...data });
      } catch (err) {
        if (alive) setState({ status: "err", message: err.message });
      }
    };
    run();
    const id = setInterval(run, 15000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const color = state.status === "ok" ? "var(--cleared)" : state.status === "err" ? "var(--flagged)" : "var(--text-faint)";

  return (
    <div className="health mono">
      <span className="health-dot" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
      {state.status === "checking" && "CONNECTING…"}
      {state.status === "ok" && `LIVE · MODEL ${state.model_loaded ? "LOADED" : "COLD"} · ${state.ms}MS`}
      {state.status === "err" && "OFFLINE"}

      <style>{`
        .health {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.72rem;
          letter-spacing: 0.06em;
          color: var(--text-dim);
        }
        .health-dot {
          width: 7px; height: 7px;
          border-radius: 50%;
          flex-shrink: 0;
        }
      `}</style>
    </div>
  );
}
