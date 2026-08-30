"""
Part C extension -- deterministic, rule-based URL risk analysis. Deliberately
NOT machine-learned: these are the same documented heuristics real phishing
filters use (IP-literal hosts, shorteners, lookalike domains, credential-
stuffing "@" tricks, homograph characters), applied to whatever URLs appear
in the inspected text. No network calls -- everything is computed from the
URL string itself, so this works offline and never leaks the pasted email
to a third party.
"""
import re
from urllib.parse import urlparse

URL_RE = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)

# SMS/smishing links are routinely written without a scheme at all (space
# constraints, and it still renders as a tappable link on most phones) --
# e.g. "bit.ly/xyz123" or "paypa1-secure.xyz/login". This catches a bare
# domain(+path), requiring a real-looking multi-char alphabetic TLD so it
# doesn't fire on version numbers ("v2.5") or sentence-ending initials
# ("e.g."). Genuinely a heuristic, not a public-suffix-list-grade parser --
# an occasional false positive on an unusual two-word phrase is the
# accepted cost of not missing the scheme-less links that dominate SMS.
BARE_DOMAIN_RE = re.compile(
    r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,24}(?:/[^\s<>"\']*)?\b'
)

IP_HOST_RE = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorte.st", "adf.ly", "bl.ink",
}

# A small, deliberately non-exhaustive set of commonly-impersonated brands --
# used only to flag *lookalikes*, not to claim completeness.
WATCHED_BRANDS = [
    "paypal", "microsoft", "apple", "amazon", "google", "netflix", "facebook",
    "instagram", "chase", "bankofamerica", "wellsfargo", "dropbox", "linkedin",
]

SUSPICIOUS_TLDS = {"xyz", "top", "click", "work", "support", "gq", "tk"}


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _label_tokens(hostname: str) -> list[str]:
    """Every '.'- and '-'-separated token in the hostname (e.g.
    'paypa1-secure.xyz' -> ['paypa1', 'secure', 'xyz']), so a brand name
    hidden in a longer, hyphenated hostname is still checked, not just the
    first full label. Deliberately simple -- a heuristic flag, not a
    public-suffix-list-grade parser."""
    return [tok for part in hostname.split(".") for tok in part.split("-") if tok]


def _has_non_ascii_homograph(hostname: str) -> bool:
    try:
        hostname.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def analyze_url(url: str) -> dict:
    signals = []
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return dict(url=url, host=None, signals=["unparseable-url"], risk_score=50)

    if not host:
        return dict(url=url, host=None, signals=["unparseable-url"], risk_score=50)

    if IP_HOST_RE.match(host):
        signals.append("ip-address-as-hostname")

    if host in SHORTENERS:
        signals.append("known-url-shortener")

    if "@" in url.split("://", 1)[-1].split("/")[0]:
        signals.append("embedded-credential-or-redirect-trick (@ in authority)")

    if host.count(".") >= 4:
        signals.append("excessive-subdomains")

    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in SUSPICIOUS_TLDS:
        signals.append(f"uncommon-tld (.{tld})")

    if _has_non_ascii_homograph(host):
        signals.append("non-ascii-characters-in-hostname (possible homograph)")

    # Legitimate iff the registrable domain (last two labels -- a
    # simplification that doesn't handle multi-part TLDs like .co.uk, but
    # this is a heuristic flag, not a public-suffix-list-grade parser) is
    # exactly "<brand>.<tld>". Anything else containing the brand's name --
    # an exact token in an unrelated domain, or a near-miss spelling -- is
    # exactly the pattern real brand-impersonation phishing uses.
    host_labels = host.split(".")
    registrable = ".".join(host_labels[-2:]) if len(host_labels) >= 2 else host
    tokens = _label_tokens(host)
    is_legitimate = any(registrable == f"{brand}.{tld}"
                         for brand in WATCHED_BRANDS
                         for tld in ("com", "net", "org", "co"))
    if not is_legitimate:
        flagged = False
        for token in tokens:
            if len(token) < 4:
                continue
            for brand in WATCHED_BRANDS:
                dist = _levenshtein(token, brand)
                if token == brand:
                    signals.append(f"brand-name-in-unrelated-domain ('{brand}' found, but domain is '{host}')")
                    flagged = True
                elif dist <= 2:
                    signals.append(f"lookalike-domain ('{token}' close to '{brand}')")
                    flagged = True
                if flagged:
                    break
            if flagged:
                break

    risk_score = min(100, len(signals) * 30)
    return dict(url=url, host=host, signals=signals, risk_score=risk_score)


def analyze_urls(text: str) -> list[dict]:
    scheme_urls = URL_RE.findall(text)
    covered_spans = [m.span() for m in URL_RE.finditer(text)]

    bare_urls = []
    for m in BARE_DOMAIN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in covered_spans):
            continue  # already matched as part of a full https?:// URL
        bare_urls.append(m.group(0))

    # de-duplicate while preserving order; bare matches are parsed with an
    # assumed scheme (analyze_url needs one to extract a hostname) but
    # reported back using the original, as-typed text.
    seen = set()
    findings = []
    for u in scheme_urls:
        if u not in seen:
            seen.add(u)
            findings.append(analyze_url(u))
    for u in bare_urls:
        if u not in seen:
            seen.add(u)
            findings.append(analyze_url(f"http://{u}") | {"url": u})
    return findings
