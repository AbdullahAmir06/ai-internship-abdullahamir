import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { fetchModels } from "../api";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] } },
};

function Row({ model, index }) {
  return (
    <motion.tr
      variants={fadeUp}
      className={model.deployed ? "row-deployed" : ""}
    >
      <td className="mono row-idx">{String(index + 1).padStart(2, "0")}</td>
      <td>
        <div className="row-name">{model.name}</div>
        <div className="row-approach">{model.approach}</div>
      </td>
      <td className="mono num">{model.test_accuracy != null ? `${(model.test_accuracy * 100).toFixed(2)}%` : "—"}</td>
      <td className="mono num">{model.test_macro_f1 != null ? model.test_macro_f1.toFixed(4) : "—"}</td>
      <td className="mono num">{model.avg_latency_ms != null ? `${model.avg_latency_ms.toFixed(2)}ms` : "—"}</td>
      <td className="mono num">{model.artifact_size || "—"}</td>
      <td>
        <span className={`status-chip ${model.deployed ? "chip-live" : "chip-file"}`}>
          {model.deployed ? "LIVE" : "ON FILE"}
        </span>
      </td>
    </motion.tr>
  );
}

export default function CaseLedger() {
  const [models, setModels] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModels().then(setModels).catch((e) => setError(e.message));
  }, []);

  return (
    <section className="ledger-section">
      <div className="ledger-inner">
        <h2>Two models, one case file.</h2>
        <p className="ledger-sub measure">
          Every figure below is a measured result from this project's own training and
          evaluation runs — not a projection. Only one model is on duty.
        </p>

        {error && <p className="mono error-text">Could not load case file: {error}</p>}

        {models && (
          <motion.div
            className="ledger-table-wrap"
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-60px" }}
            transition={{ staggerChildren: 0.12 }}
          >
            <table className="ledger-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Model</th>
                  <th>Test acc.</th>
                  <th>Macro F1</th>
                  <th>Latency</th>
                  <th>Size</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => <Row model={m} index={i} key={m.name} />)}
              </tbody>
            </table>
          </motion.div>
        )}

        {models && (
          <div className="ledger-notes">
            {models.map((m) => m.note && (
              <p key={m.name} className="ledger-note">
                <strong>{m.name.split(":")[0]}</strong> — {m.note}
              </p>
            ))}
          </div>
        )}
      </div>

      <style>{`
        .ledger-section { padding: 7rem 6vw; background: var(--ink-2); }
        .ledger-inner { max-width: 1180px; margin: 0 auto; }
        .ledger-section h2 { font-size: clamp(2rem, 4vw, 3rem); max-width: 18ch; }
        .ledger-sub { margin-top: 1.1rem; color: var(--text-dim); font-size: 1.05rem; }

        .ledger-table-wrap {
          margin-top: 2.5rem;
          overflow-x: auto;
          border: 1px solid var(--panel-border);
          border-radius: 6px;
        }
        .ledger-table { width: 100%; border-collapse: collapse; min-width: 720px; }
        .ledger-table th {
          text-align: left;
          font-family: var(--font-mono);
          font-size: 0.7rem;
          letter-spacing: 0.08em;
          color: var(--text-faint);
          font-weight: 400;
          padding: 0.9rem 1rem;
          border-bottom: 1px solid var(--panel-border);
          background: var(--panel);
        }
        .ledger-table td {
          padding: 1rem;
          border-bottom: 1px solid var(--panel-border);
          vertical-align: middle;
        }
        .ledger-table tbody tr:last-child td { border-bottom: none; }
        .row-deployed { background: linear-gradient(90deg, rgba(234,167,62,0.08), transparent 40%); }
        .row-idx { color: var(--text-faint); width: 2.5rem; }
        .row-name { font-weight: 500; }
        .row-approach { color: var(--text-dim); font-size: 0.85rem; margin-top: 0.2rem; }
        .num { color: var(--text); }

        .status-chip {
          font-family: var(--font-mono);
          font-size: 0.68rem;
          letter-spacing: 0.06em;
          padding: 0.25rem 0.55rem;
          border-radius: 3px;
          border: 1px solid;
        }
        .chip-live { color: var(--cleared); border-color: var(--cleared); background: var(--cleared-dim); }
        .chip-file { color: var(--text-dim); border-color: var(--panel-border); }

        .ledger-notes { margin-top: 2rem; display: flex; flex-direction: column; gap: 0.6rem; }
        .ledger-note { color: var(--text-dim); font-size: 0.92rem; max-width: 72ch; }
        .ledger-note strong { color: var(--text); }

        .error-text { color: var(--flagged); margin-top: 1.5rem; }
      `}</style>
    </section>
  );
}
