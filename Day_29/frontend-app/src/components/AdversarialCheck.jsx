import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldAlert } from "lucide-react";
import { adversarialCheck } from "../api";

// A real evasion-resistance probe: applies leetspeak to the SAME text just
// inspected and genuinely re-runs Model A on it (backend/app/adversarial.py)
// -- both verdicts shown here are real model calls, not a scripted demo.
export default function AdversarialCheck({ text }) {
  const [state, setState] = useState("idle"); // idle | checking | done | error
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  const run = async () => {
    setState("checking");
    try {
      const data = await adversarialCheck(text);
      setResult(data);
      setState("done");
    } catch (e) {
      setErr(e.message);
      setState("error");
    }
  };

  return (
    <div className="adv">
      <button className="adv-btn mono" onClick={run} disabled={state === "checking"}>
        <ShieldAlert size={14} strokeWidth={2} />
        {state === "checking" ? "TESTING EVASION…" : "TEST EVASION RESISTANCE"}
      </button>

      <AnimatePresence>
        {state === "done" && result && (
          <motion.div
            className="adv-result"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <p className="adv-note">
              Same email, with common trigger words swapped for leetspeak
              (a real technique used to dodge keyword filters):
            </p>
            <p className="adv-perturbed mono">{result.perturbed_text}</p>
            <div className="adv-row mono">
              <span>ORIGINAL</span>
              <span>{result.original_label.toUpperCase()} · {(result.original_confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="adv-row mono">
              <span>PERTURBED</span>
              <span>{result.perturbed_label.toUpperCase()} · {(result.perturbed_confidence * 100).toFixed(1)}%</span>
            </div>
            <p className={`adv-verdict mono ${result.verdict_flipped ? "adv-verdict-flip" : "adv-verdict-hold"}`}>
              {result.verdict_flipped
                ? "VERDICT FLIPPED — this evasion technique fooled the model."
                : "VERDICT HELD — model stayed correct despite the evasion attempt."}
            </p>
          </motion.div>
        )}
        {state === "error" && (
          <motion.p className="adv-error mono" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            CHECK FAILED — {err}
          </motion.p>
        )}
      </AnimatePresence>

      <style>{`
        .adv { margin-top: 1rem; }
        .adv-btn {
          display: inline-flex; align-items: center; gap: 0.5rem;
          background: none; border: 1px solid var(--panel-border); color: var(--text-dim);
          padding: 0.5rem 0.9rem; border-radius: 4px;
          font-size: 0.72rem; letter-spacing: 0.05em;
        }
        .adv-btn:hover:not(:disabled) { border-color: var(--lamp-dim); color: var(--lamp-bright); }
        .adv-btn:disabled { opacity: 0.6; cursor: not-allowed; }

        .adv-result { overflow: hidden; margin-top: 0.9rem; }
        .adv-note { font-size: 0.82rem; color: var(--text-dim); margin-bottom: 0.6rem; }
        .adv-perturbed {
          font-size: 0.8rem; color: var(--text); background: var(--panel);
          border: 1px solid var(--panel-border); border-radius: 4px;
          padding: 0.7rem 0.9rem; margin-bottom: 0.8rem; line-height: 1.5;
        }
        .adv-row {
          display: flex; justify-content: space-between;
          font-size: 0.78rem; color: var(--text-dim);
          padding: 0.35rem 0; border-top: 1px solid var(--panel-border);
        }
        .adv-verdict { font-size: 0.78rem; margin-top: 0.8rem; padding-top: 0.6rem; }
        .adv-verdict-flip { color: var(--flagged); }
        .adv-verdict-hold { color: var(--cleared); }
        .adv-error { color: var(--flagged); font-size: 0.8rem; margin-top: 0.6rem; }
      `}</style>
    </div>
  );
}
