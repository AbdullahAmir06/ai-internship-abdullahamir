export default function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="footer-title">Phishing Email Inspection Desk</div>
      </div>
      <style>{`
        .site-footer {
          padding: 3rem 6vw;
          background: var(--ink-2);
          border-top: 1px solid var(--panel-border);
        }
        .footer-inner {
          max-width: 1180px; margin: 0 auto;
        }
        .footer-title { font-family: var(--font-display); font-weight: 500; font-size: 1.05rem; }
      `}</style>
    </footer>
  );
}
