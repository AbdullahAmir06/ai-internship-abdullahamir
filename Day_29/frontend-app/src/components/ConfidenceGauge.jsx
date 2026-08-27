import { motion } from "framer-motion";

// Donated discipline (from the declined particle-collider challenger):
// confidence is real visual weight -- fill width, brightness, and glow all
// scale off the same number -- never an inert percentage sitting next to a
// bar for decoration.
export default function ConfidenceGauge({ confidence, color }) {
  const pct = Math.round(confidence * 100);
  return (
    <div className="gauge">
      <div className="gauge-track">
        <motion.div
          className="gauge-fill"
          style={{ background: color, boxShadow: `0 0 ${8 + pct / 5}px ${color}` }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.3 }}
        />
        {[20, 40, 60, 80].map((tick) => (
          <span key={tick} className="gauge-tick" style={{ left: `${tick}%` }} />
        ))}
      </div>
      <span className="mono gauge-readout" style={{ color }}>
        {pct.toString().padStart(2, "0")}%
      </span>

      <style>{`
        .gauge { display: flex; align-items: center; gap: 0.9rem; width: 100%; }
        .gauge-track {
          position: relative;
          flex: 1;
          height: 10px;
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-radius: 2px;
          overflow: hidden;
        }
        .gauge-fill { height: 100%; border-radius: 1px; }
        .gauge-tick {
          position: absolute;
          top: 0; bottom: 0;
          width: 1px;
          background: rgba(0,0,0,0.35);
        }
        .gauge-readout {
          font-size: 1.05rem;
          font-weight: 500;
          min-width: 3.2ch;
          text-align: right;
        }
      `}</style>
    </div>
  );
}
