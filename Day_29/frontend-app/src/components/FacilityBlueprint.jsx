export default function FacilityBlueprint() {
  return (
    <section className="blueprint-section">
      <div className="blueprint-inner">
        <h2>The facility floor plan.</h2>
        <div className="blueprint-frame">
          <img src="/architecture.png" alt="System architecture diagram" />
        </div>
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
