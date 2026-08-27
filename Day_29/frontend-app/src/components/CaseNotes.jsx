import { motion } from "framer-motion";

const fade = (delay = 0) => ({
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] } },
});

export default function CaseNotes() {
  return (
    <section className="notes-section">
      <div className="notes-inner">
        <motion.div
          className="notes-tab mono"
          initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={fade()}
        >
          POLICY NOTE
        </motion.div>

        <motion.h2
          initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={fade(0.05)}
        >
          Why only one model is on duty.
        </motion.h2>

        <div className="notes-grid">
          <motion.div
            className="notes-col"
            initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={fade(0.1)}
          >
            <span className="notes-num mono">01</span>
            <p className="measure">
              An earlier build in this same body of work put a full Transformer-based service
              on a free-tier host with a hard 512MB memory ceiling. It measured roughly 695MB
              at runtime — a real, logged out-of-memory failure, not a theoretical concern.
              Two "obvious" fixes were tried and measured directly: lower-precision weight
              loading, and dynamic quantization. Both made memory usage <em>worse</em>, not
              better.
            </p>
          </motion.div>

          <motion.div
            className="notes-col"
            initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={fade(0.18)}
          >
            <span className="notes-num mono">02</span>
            <p className="measure">
              That finding set this project's architecture before a single email was
              classified: the fine-tuned Transformer (Model B) is trained and evaluated in
              full, on the record, so its real accuracy is known and comparable — but it
              never touches the live service. The classical model (Model A) does the actual
              inspecting.
            </p>
          </motion.div>

          <motion.div
            className="notes-col"
            initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={fade(0.26)}
          >
            <span className="notes-num mono">03</span>
            <p className="measure">
              This isn't caution for its own sake — it's the same measured-proof discipline
              applied on purpose. The deployment certificate below is that proof, run again
              for this project specifically, under the same hard memory cap that failed
              before.
            </p>
          </motion.div>
        </div>
      </div>

      <style>{`
        .notes-section { padding: 7rem 6vw; background: var(--ink); }
        .notes-inner { max-width: 1180px; margin: 0 auto; }
        .notes-tab {
          display: inline-block;
          font-size: 0.7rem; letter-spacing: 0.12em;
          color: var(--lamp-bright);
          border: 1px solid var(--lamp-dim);
          padding: 0.3rem 0.7rem;
          border-radius: 3px;
          margin-bottom: 1.2rem;
        }
        .notes-section h2 { font-size: clamp(2rem, 4vw, 3rem); max-width: 18ch; }
        .notes-grid {
          margin-top: 3rem;
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 2.5rem;
        }
        .notes-col { border-top: 1px solid var(--panel-border); padding-top: 1.2rem; }
        .notes-num { display: block; color: var(--lamp-dim); font-size: 0.85rem; margin-bottom: 0.7rem; }
        .notes-col p { color: var(--text-dim); font-size: 0.98rem; }
        .notes-col em { color: var(--text); font-style: normal; }

        @media (max-width: 860px) {
          .notes-grid { grid-template-columns: 1fr; gap: 2rem; }
        }
      `}</style>
    </section>
  );
}
