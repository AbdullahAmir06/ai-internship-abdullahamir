import { motion } from "framer-motion";

export default function FacilityBlueprint() {
  return (
    <section className="blueprint-section">
      <div className="blueprint-inner">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          The facility floor plan.
        </motion.h2>
        <motion.div
          className="blueprint-frame"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        >
          <img src="/architecture.png" alt="System architecture diagram" />
        </motion.div>
      </div>

      <style>{`
        .blueprint-section { padding: 7rem 6vw; background: var(--ink); }
        .blueprint-inner { max-width: 1180px; margin: 0 auto; }
        .blueprint-section h2 { font-size: clamp(2rem, 4vw, 3rem); }
        .blueprint-frame {
          margin-top: 2.5rem;
          border: 1px solid var(--panel-border);
          border-radius: 8px;
          padding: 1.5rem;
          background: #f4f1e8;
        }
        .blueprint-frame img { width: 100%; display: block; border-radius: 4px; }
      `}</style>
    </section>
  );
}
