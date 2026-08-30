import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search } from "lucide-react";
import { predict } from "../api";
import ConfidenceGauge from "./ConfidenceGauge";
import HealthBadge from "./HealthBadge";
import HighlightedText from "./HighlightedText";
import UrlFindings from "./UrlFindings";
import AdversarialCheck from "./AdversarialCheck";

const SAMPLES = {
  email: "Dear Customer, we detected unusual activity on your account. " +
    "Verify your identity within 24 hours or your access will be suspended. " +
    "Click here to confirm your details immediately.",
  sms: "URGENT: Your bank account has been locked. Verify now: bit.ly/xyz123",
};

const CHANNEL_COPY = {
  email: { noun: "email", exhibit: "EMAIL TEXT", placeholder: "Paste the email body here…" },
  sms: { noun: "text message", exhibit: "SMS TEXT", placeholder: "Paste the SMS text here…" },
};

export default function InspectionHero() {
  const [channel, setChannel] = useState("email");
  const [text, setText] = useState("");
  const [phase, setPhase] = useState("idle"); // idle | scanning | done | error
  const [result, setResult] = useState(null);
  const [errMsg, setErrMsg] = useState("");
  const [modelBAvailable, setModelBAvailable] = useState(false);
  const [selectedModel, setSelectedModel] = useState("a");

  const switchChannel = (next) => {
    if (next === channel || phase === "scanning") return;
    setChannel(next);
    setText("");
    setResult(null);
    setPhase("idle");
    if (next === "sms") setSelectedModel("a"); // Model B is email-only
  };

  const inspect = async () => {
    if (!text.trim()) return;
    setPhase("scanning");
    setResult(null);
    const t0 = performance.now();
    try {
      const data = await predict(text, selectedModel, channel);
      const elapsed = performance.now() - t0;
      // The sweep is the signature motion -- give it room to read even
      // when the API itself answers in a few milliseconds. Model B's
      // first call also has to load weights from disk (a few seconds),
      // which already dwarfs the sweep, so this only matters for Model A.
      const minSweep = 900;
      if (elapsed < minSweep) await new Promise((r) => setTimeout(r, minSweep - elapsed));
      setResult(data);
      setPhase("done");
    } catch (err) {
      setErrMsg(err.message);
      setPhase("error");
    }
  };

  const verdictColor = result?.label === "phishing" ? "var(--flagged)" : "var(--cleared)";

  return (
    <section className="hero">
      <div className="hero-beam" aria-hidden="true" />
      <div className="hero-inner">
        <div className="hero-top">
          <HealthBadge onHealth={(d) => setModelBAvailable(!!d.model_b_available)} />
        </div>

        <h1 className="hero-title">
          Every {CHANNEL_COPY[channel].noun} <span className="hero-title-accent">tells on itself</span>
          <br />under the right light.
        </h1>
        <p className="hero-sub measure">
          This is case intake for a suspicious {CHANNEL_COPY[channel].noun}. Paste the text
          below and a trained model inspects it, then returns an evidence-backed verdict —
          safe or phishing — with a real, measured confidence score. No account, no upload,
          nothing stored.
        </p>

        <div className="channel-tabs" role="tablist" aria-label="Channel">
          <button
            role="tab" aria-selected={channel === "email"} className="channel-tab mono"
            onClick={() => switchChannel("email")}
          >
            EMAIL
          </button>
          <button
            role="tab" aria-selected={channel === "sms"} className="channel-tab mono"
            onClick={() => switchChannel("sms")}
          >
            SMS
          </button>
        </div>

        <div className="desk">
          <div className="document-panel">
            <div className="document-header mono">
              <span>EXHIBIT — {CHANNEL_COPY[channel].exhibit}</span>
              <div className="header-actions">
                {channel === "email" && modelBAvailable && (
                  <div className="model-picker" role="radiogroup" aria-label="Model">
                    <button
                      className={`model-btn ${selectedModel === "a" ? "model-btn-active" : ""}`}
                      onClick={() => setSelectedModel("a")}
                      disabled={phase === "scanning"}
                      aria-pressed={selectedModel === "a"}
                    >
                      MODEL A
                    </button>
                    <button
                      className={`model-btn ${selectedModel === "b" ? "model-btn-active" : ""}`}
                      onClick={() => setSelectedModel("b")}
                      disabled={phase === "scanning"}
                      aria-pressed={selectedModel === "b"}
                    >
                      MODEL B
                    </button>
                  </div>
                )}
                <button className="sample-btn mono" onClick={() => setText(SAMPLES[channel])}>
                  load sample
                </button>
              </div>
            </div>
            <textarea
              className="document-textarea"
              placeholder={CHANNEL_COPY[channel].placeholder}
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={6}
              disabled={phase === "scanning"}
            />

            <AnimatePresence>
              {phase === "scanning" && (
                <motion.div
                  className="sweep"
                  initial={{ x: "-110%" }}
                  animate={{ x: "110%" }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 1.1, ease: [0.65, 0, 0.35, 1] }}
                />
              )}
            </AnimatePresence>

            <div className="document-footer">
              <button
                className="inspect-btn mono"
                onClick={inspect}
                disabled={phase === "scanning" || !text.trim()}
              >
                <Search size={16} strokeWidth={2} />
                {phase === "scanning"
                  ? (selectedModel === "b" ? "INSPECTING (MODEL B)…" : "INSPECTING…")
                  : "INSPECT"}
              </button>

              <AnimatePresence mode="wait">
                {phase === "done" && result && (
                  <motion.div
                    key="stamp"
                    className="stamp"
                    style={{ color: verdictColor, borderColor: verdictColor }}
                    initial={{ opacity: 0, scale: 1.6, rotate: -8 }}
                    animate={{ opacity: 1, scale: 1, rotate: -6 }}
                    transition={{ type: "spring", stiffness: 340, damping: 14 }}
                  >
                    {result.label === "phishing" ? "FLAGGED" : "CLEARED"}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {phase === "error" && (
              <p className="error-text mono">INSPECTION FAILED — {errMsg}</p>
            )}

            {phase === "done" && result && (
              <div className="annotated">
                <div className="annotated-label mono">ANNOTATED COPY — MODEL'S OWN WEIGHTS</div>
                <p className="annotated-text">
                  <HighlightedText text={text} highlights={result.highlights} />
                </p>
                <UrlFindings findings={result.url_findings} />
              </div>
            )}
          </div>

          <AnimatePresence>
            {phase === "done" && result && (
              <motion.div
                className="verdict-card"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.15, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              >
                <div className="verdict-row mono">
                  <span>VERDICT</span>
                  <span style={{ color: verdictColor }}>{result.label.toUpperCase()}</span>
                </div>
                <ConfidenceGauge confidence={result.confidence} color={verdictColor} />
                <div className="verdict-row mono verdict-latency">
                  <span>LATENCY</span>
                  <span>{result.latency_ms.toFixed(2)}ms</span>
                </div>
                <div className="verdict-row mono">
                  <span>SERVED BY</span>
                  <span>
                    {result.model === "b" ? "MODEL B (local)" : `MODEL A (${result.channel}, live)`}
                  </span>
                </div>
                {result.model === "a" && <AdversarialCheck text={text} channel={channel} />}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <style>{`
        .hero {
          position: relative;
          overflow: hidden;
          padding: 5.5rem 6vw 4.5rem;
          min-height: 100svh;
          display: flex;
          align-items: center;
        }
        .hero-beam {
          position: absolute;
          inset: -20% -10%;
          background: linear-gradient(115deg,
            transparent 0%, transparent 38%,
            rgba(234,167,62,0.16) 47%,
            rgba(246,192,101,0.30) 52%,
            rgba(234,167,62,0.16) 57%,
            transparent 66%, transparent 100%);
          pointer-events: none;
        }
        .hero-inner { position: relative; width: 100%; max-width: 1180px; margin: 0 auto; }
        .hero-top {
          display: flex; justify-content: flex-end; align-items: center;
          margin-bottom: 2.6rem;
        }
        .hero-title {
          font-size: clamp(2.6rem, 6vw, 5rem);
          max-width: 16ch;
        }
        .hero-title-accent { color: var(--lamp); }
        .hero-sub {
          margin-top: 1.5rem;
          font-size: 1.15rem;
          color: var(--text-dim);
        }

        .channel-tabs {
          display: inline-flex;
          margin-top: 2rem;
          border: 1px solid var(--panel-border);
          border-radius: 4px;
          overflow: hidden;
        }
        .channel-tab {
          background: var(--panel);
          border: none;
          color: var(--text-dim);
          padding: 0.55rem 1.4rem;
          font-size: 0.78rem;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: background 0.15s ease, color 0.15s ease;
        }
        .channel-tab + .channel-tab { border-left: 1px solid var(--panel-border); }
        .channel-tab[aria-selected="true"] { background: var(--lamp); color: #241a08; }
        .channel-tab:not([aria-selected="true"]):hover { color: var(--lamp-bright); }

        .desk {
          margin-top: 2rem;
          display: grid;
          grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
          gap: 1.5rem;
          align-items: start;
        }
        .document-panel {
          position: relative;
          background: var(--paper);
          color: #2a2013;
          border-radius: 6px;
          padding: 1.4rem 1.5rem 1.6rem;
          box-shadow: 0 30px 60px -20px rgba(0,0,0,0.55), 0 2px 0 rgba(0,0,0,0.15);
          overflow: hidden;
        }
        .document-header {
          display: flex; justify-content: space-between; align-items: center;
          font-size: 0.72rem; letter-spacing: 0.08em;
          color: #6b5c3c;
          margin-bottom: 0.8rem;
        }
        .header-actions { display: flex; align-items: center; gap: 0.6rem; }
        .model-picker {
          display: inline-flex;
          border: 1px solid #6b5c3c55;
          border-radius: 3px;
          overflow: hidden;
        }
        .model-btn {
          background: none; border: none; color: #6b5c3c;
          padding: 0.2rem 0.55rem; font-size: 0.66rem; letter-spacing: 0.05em;
          font-family: var(--font-mono); cursor: pointer;
        }
        .model-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .model-btn-active { background: #2a2013; color: var(--paper-2); }
        .sample-btn {
          background: none; border: 1px solid #6b5c3c55; color: #6b5c3c;
          padding: 0.2rem 0.6rem; border-radius: 3px; font-size: 0.68rem;
          letter-spacing: 0.06em;
        }
        .sample-btn:hover { border-color: #6b5c3c; }
        .document-textarea {
          width: 100%;
          background: var(--paper-2);
          border: 1px solid #b8a878;
          border-radius: 4px;
          padding: 0.9rem 1rem;
          font-family: var(--font-body);
          font-size: 0.98rem;
          line-height: 1.55;
          color: #2a2013;
          resize: vertical;
          min-height: 130px;
        }
        .document-textarea::placeholder { color: #8a7a54; }
        .document-textarea:focus-visible { outline: 2px solid var(--lamp-dim); outline-offset: 1px; }

        .sweep {
          position: absolute;
          top: 0; bottom: 0; width: 32%;
          background: linear-gradient(90deg,
            transparent, rgba(255,255,255,0.55), transparent);
          mix-blend-mode: soft-light;
          pointer-events: none;
        }

        .document-footer {
          display: flex; align-items: center; justify-content: space-between;
          margin-top: 1rem; min-height: 2.6rem;
        }
        .inspect-btn {
          display: inline-flex; align-items: center; gap: 0.5rem;
          background: var(--lamp);
          color: #241a08;
          border: none; border-radius: 4px;
          padding: 0.7rem 1.3rem;
          font-size: 0.82rem; font-weight: 500; letter-spacing: 0.06em;
          transition: transform 0.15s ease, background 0.15s ease;
        }
        .inspect-btn:hover:not(:disabled) { background: var(--lamp-bright); transform: translateY(-1px); }
        .inspect-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .stamp {
          font-family: var(--font-display);
          font-weight: 700;
          font-size: 1.5rem;
          letter-spacing: 0.06em;
          border: 3px solid;
          border-radius: 4px;
          padding: 0.3rem 0.9rem;
          transform: rotate(-6deg);
        }

        .error-text {
          color: var(--flagged); font-size: 0.8rem; margin-top: 0.8rem;
        }

        .annotated {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid #b8a87855;
        }
        .annotated-label {
          font-size: 0.68rem; letter-spacing: 0.08em; color: #6b5c3c;
          margin-bottom: 0.6rem;
        }
        .annotated-text {
          font-size: 0.98rem; line-height: 1.6; color: #2a2013;
          margin: 0;
        }

        .verdict-card {
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-radius: 6px;
          padding: 1.3rem 1.4rem;
          display: flex; flex-direction: column; gap: 1rem;
          align-self: stretch;
        }
        .verdict-row {
          display: flex; justify-content: space-between; align-items: center;
          font-size: 0.78rem; letter-spacing: 0.06em; color: var(--text-dim);
        }
        .verdict-latency { padding-top: 0.6rem; border-top: 1px solid var(--panel-border); color: var(--text-faint); }

        @media (max-width: 860px) {
          .desk { grid-template-columns: 1fr; }
          .hero { padding: 4rem 5vw 3rem; }
        }
        @media (max-width: 560px) {
          .hero-top { flex-direction: column; align-items: flex-start; gap: 0.7rem; }
        }
      `}</style>
    </section>
  );
}
