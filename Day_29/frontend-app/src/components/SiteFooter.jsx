function SourceMark() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="8 4 4 12 8 20" />
      <polyline points="16 4 20 12 16 20" />
    </svg>
  );
}

export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div>
          <div className="footer-title">Phishing Email Inspection Desk</div>
          <p className="mono footer-meta">PKCERT AI &amp; Software Development Internship — Capstone</p>
        </div>
        <a
          className="footer-link mono"
          href="https://github.com/AbdullahAmir06/ai-internship-abdullahamir"
          target="_blank" rel="noreferrer"
        >
          <SourceMark />
          source
        </a>
      </div>
      <style>{`
        .site-footer {
          padding: 3rem 6vw;
          background: var(--ink-2);
          border-top: 1px solid var(--panel-border);
        }
        .footer-inner {
          max-width: 1180px; margin: 0 auto;
          display: flex; justify-content: space-between; align-items: center;
          flex-wrap: wrap; gap: 1rem;
        }
        .footer-title { font-family: var(--font-display); font-weight: 500; font-size: 1.05rem; }
        .footer-meta { color: var(--text-faint); font-size: 0.75rem; margin-top: 0.3rem; }
        .footer-link {
          display: inline-flex; align-items: center; gap: 0.5rem;
          color: var(--text-dim); font-size: 0.8rem; text-decoration: none;
          border: 1px solid var(--panel-border); padding: 0.5rem 0.9rem; border-radius: 5px;
        }
        .footer-link:hover { color: var(--lamp-bright); border-color: var(--lamp-dim); }
      `}</style>
    </footer>
  );
}
