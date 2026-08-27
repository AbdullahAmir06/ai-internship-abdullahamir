import { motion } from "framer-motion";

const line = {
  hidden: { scaleX: 0 },
  show: { scaleX: 1, transition: { duration: 0.7, ease: [0.65, 0, 0.35, 1] } },
};
const rise = (delay = 0) => ({
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: 0.5, delay, ease: "easeOut" } },
});

export default function CaseNotes() {
  return (
    <section className="notes-section">
      <div className="notes-inner">
        <h2>Why only one model is on duty.</h2>

        <div className="notes-grid">
          <div className="notes-col">
            <motion.span
              className="notes-rule"
              initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={line}
            />
            <motion.p
              className="measure"
              initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={rise(0.35)}
            >
              An earlier build in this same body of work put a full Transformer-based service
              on a free-tier host with a hard 512MB memory ceiling. It measured roughly 695MB
              at runtime — a real, logged out-of-memory failure, not a theoretical concern.
              Two "obvious" fixes were tried and measured directly: lower-precision weight
              loading, and dynamic quantization. Both made memory usage <em>worse</em>, not
              better.
            </motion.p>
          </div>

          <div className="notes-col">
            <motion.span
              className="notes-rule"
              initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={line}
              transition={{ delay: 0.12 }}
            />
            <motion.p
              className="measure"
              initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={rise(0.47)}
            >
              That finding set this project's architecture before a single email was
              classified: the fine-tuned Transformer (Model B) is trained and evaluated in
              full, on the record, so its real accuracy is known and comparable — but it
              never touches the live service. The classical model (Model A) does the actual
              inspecting.
            </motion.p>
          </div>

          <div className="notes-col">
            <motion.span
              className="notes-rule"
              initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={line}
              transition={{ delay: 0.24 }}
            />
            <motion.p
              className="measure"
              initial="hidden" whileInView="show" viewport={{ once: true, margin: "-80px" }} variants={rise(0.59)}
            >
              This isn't caution for its own sake — it's the same measured-proof discipline
              applied on purpose. The deployment certificate below is that proof, run again
              for this project specifically, under the same hard memory cap that failed
              before.
            </motion.p>
          </div>
        </div>
      </div>

      <style>{`
        .notes-section { padding: 7rem 6vw; background: var(--ink); }
        .notes-inner { max-width: 1180px; margin: 0 auto; }
        .notes-section h2 { font-size: clamp(2rem, 4vw, 3rem); max-width: 18ch; }
        .notes-grid {
          margin-top: 3rem;
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 2.5rem;
        }
        .notes-col { padding-top: 1.2rem; }
        .notes-rule {
          display: block;
          height: 1px;
          background: var(--lamp-dim);
          transform-origin: left;
          margin-bottom: 1.2rem;
        }
        .notes-col p { color: var(--text-dim); font-size: 0.98rem; }
        .notes-col em { color: var(--text); font-style: normal; }

        @media (max-width: 860px) {
          .notes-grid { grid-template-columns: 1fr; gap: 2rem; }
        }
      `}</style>
    </section>
  );
}
