// Renders the inspected text with the model's own real contribution spans
// marked -- never a decorative highlight, always the exact char offsets
// the backend computed from its actual TF-IDF+LogisticRegression weights.
export default function HighlightedText({ text, highlights }) {
  if (!highlights || highlights.length === 0) {
    return <span>{text}</span>;
  }

  const segments = [];
  let cursor = 0;
  for (const h of highlights) {
    if (h.start > cursor) segments.push({ text: text.slice(cursor, h.start), mark: null });
    segments.push({ text: text.slice(h.start, h.end), mark: h.direction, weight: h.weight });
    cursor = h.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), mark: null });

  return (
    <>
      {segments.map((seg, i) =>
        seg.mark ? (
          <mark
            key={i}
            className={`hl hl-${seg.mark}`}
            title={`${seg.mark === "phishing" ? "pushes toward phishing" : "pushes toward safe"} (weight ${seg.weight})`}
          >
            {seg.text}
          </mark>
        ) : (
          <span key={i}>{seg.text}</span>
        )
      )}
      <style>{`
        .hl { border-radius: 3px; padding: 0 0.1em; color: inherit; }
        .hl-phishing { background: rgba(200, 65, 44, 0.25); box-shadow: inset 0 -1.5px 0 var(--flagged); }
        .hl-safe { background: rgba(47, 143, 104, 0.22); box-shadow: inset 0 -1.5px 0 var(--cleared); }
      `}</style>
    </>
  );
}
