// Real, rule-based URL risk signals (see backend/app/url_analysis.py) --
// never machine-learned, never fabricated: every signal shown here is a
// deterministic check against the actual URL string the user pasted.
export default function UrlFindings({ findings }) {
  if (!findings || findings.length === 0) return null;

  return (
    <div className="url-findings">
      <div className="url-findings-label mono">LINKS IN EVIDENCE</div>
      {findings.map((f, i) => (
        <div className="url-row" key={i}>
          <div className="url-row-top">
            <span className="mono url-text">{f.url}</span>
            <span className={`url-risk mono ${f.signals.length > 0 ? "url-risk-flagged" : "url-risk-clear"}`}>
              {f.signals.length > 0 ? `RISK ${f.risk_score}` : "CLEAR"}
            </span>
          </div>
          {f.signals.length > 0 && (
            <ul className="url-signals">
              {f.signals.map((s, j) => <li key={j} className="mono">{s}</li>)}
            </ul>
          )}
        </div>
      ))}

      <style>{`
        .url-findings { margin-top: 1.4rem; padding-top: 1rem; border-top: 1px solid #b8a87855; }
        .url-findings-label { font-size: 0.68rem; letter-spacing: 0.08em; color: #6b5c3c; margin-bottom: 0.6rem; }
        .url-row { margin-bottom: 0.6rem; }
        .url-row-top { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; flex-wrap: wrap; }
        .url-text { font-size: 0.78rem; color: #2a2013; word-break: break-all; }
        .url-risk { font-size: 0.68rem; letter-spacing: 0.05em; padding: 0.1rem 0.4rem; border-radius: 3px; flex-shrink: 0; }
        .url-risk-flagged { background: rgba(200,65,44,0.18); color: var(--flagged); }
        .url-risk-clear { background: rgba(47,143,104,0.16); color: var(--cleared); }
        .url-signals { margin: 0.3rem 0 0; padding-left: 1.1rem; }
        .url-signals li { font-size: 0.72rem; color: #6b5c3c; margin-bottom: 0.15rem; }
      `}</style>
    </div>
  );
}
