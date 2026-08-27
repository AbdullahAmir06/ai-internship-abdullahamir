import { motion } from "framer-motion";
import { ShieldCheck } from "lucide-react";

// Numbers here are filled in from an actual `docker stats` run against this
// project's own built image -- never estimated. See FINAL_REPORT.md for the
// full command transcript.
const MEMORY_USED_MIB = 110;
const MEMORY_LIMIT_MIB = 512;
const MEMORY_PCT = "21.48";
const IMAGE_SIZE = "649MB";
const LIVE_URL = "day29-phishing-inspector.onrender.com";

export default function DeploymentCertificate() {
  return (
    <section className="cert-section">
      <div className="cert-inner">
        <motion.div
          className="cert-card"
          initial={{ opacity: 0, scale: 0.97 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.5, ease: [0.34, 1.2, 0.64, 1] }}
        >
          <h2>
            <ShieldCheck size={26} strokeWidth={2} color="var(--cleared)" className="cert-h2-icon" />
            Verified: cleared for the same limit that failed before.
          </h2>
          <p className="measure cert-sub">
            Run under a hard <code className="mono">--memory=512m</code> cap — the exact
            ceiling that caused the earlier build's out-of-memory failure — and verified
            live, not just built.
          </p>

          <div className="cert-grid">
            <div className="cert-stat">
              <span className="cert-stat-label mono">CONTAINER MEMORY</span>
              <span className="cert-stat-value mono">
                {MEMORY_USED_MIB}<span className="cert-stat-unit">MiB</span>
              </span>
              <span className="cert-stat-detail mono">of {MEMORY_LIMIT_MIB}MiB cap · {MEMORY_PCT}%</span>
            </div>
            <div className="cert-stat">
              <span className="cert-stat-label mono">IMAGE SIZE</span>
              <span className="cert-stat-value mono">{IMAGE_SIZE}</span>
              <span className="cert-stat-detail mono">no torch/transformers at runtime</span>
            </div>
            <div className="cert-stat">
              <span className="cert-stat-label mono">LIVE ENDPOINT</span>
              <span className="cert-stat-value cert-live mono">
                <a href={`https://${LIVE_URL}`} target="_blank" rel="noreferrer">{LIVE_URL}</a>
              </span>
              <span className="cert-stat-detail mono">Render, free tier</span>
            </div>
          </div>

          <pre className="cert-terminal mono">
{`$ curl https://${LIVE_URL}/healthz
{"status":"ok","model_loaded":true,"uptime_s":...}`}
          </pre>
        </motion.div>
      </div>

      <style>{`
        .cert-section { padding: 7rem 6vw; background: var(--ink-2); }
        .cert-inner { max-width: 1180px; margin: 0 auto; }
        .cert-card {
          background: var(--panel);
          border: 1px solid var(--cleared-dim);
          border-radius: 8px;
          padding: 3rem clamp(1.5rem, 4vw, 3.5rem);
        }
        .cert-card h2 {
          font-size: clamp(1.8rem, 3.4vw, 2.6rem);
          max-width: 22ch;
          display: flex;
          align-items: center;
          gap: 0.7rem;
        }
        .cert-h2-icon { flex-shrink: 0; }
        .cert-sub { margin-top: 1rem; color: var(--text-dim); }
        .cert-sub code { color: var(--lamp-bright); background: var(--ink); padding: 0.1rem 0.4rem; border-radius: 3px; }

        .cert-grid {
          margin-top: 2.5rem;
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1.5rem;
        }
        .cert-stat {
          border-top: 1px solid var(--panel-border);
          padding-top: 1rem;
          display: flex; flex-direction: column; gap: 0.3rem;
        }
        .cert-stat-label { font-size: 0.68rem; letter-spacing: 0.08em; color: var(--text-faint); }
        .cert-stat-value {
          font-size: clamp(1.1rem, 4.5vw, 1.9rem);
          color: var(--cleared);
          font-weight: 500;
          overflow-wrap: anywhere;
        }
        .cert-stat-unit { font-size: 1.1rem; margin-left: 0.15rem; color: var(--text-dim); }
        .cert-stat-detail { font-size: 0.78rem; color: var(--text-dim); }
        .cert-live { font-size: 1.05rem; word-break: break-all; }
        .cert-live a { color: var(--cleared); }

        .cert-terminal {
          margin-top: 2.2rem;
          background: var(--ink);
          border: 1px solid var(--panel-border);
          border-radius: 6px;
          padding: 1rem 1.2rem;
          font-size: 0.82rem;
          color: var(--text-dim);
          overflow-x: auto;
        }

        @media (max-width: 780px) {
          .cert-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </section>
  );
}
