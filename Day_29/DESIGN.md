---
name: Phishing Email Inspection Desk
description: A document-authentication-checkpoint system for live phishing detection — case-file dark ground, one dominant inspection-lamp amber, strict crimson/teal verdict semantics.
colors:
  case-file-ink: "#15120e"
  case-file-ink-2: "#1c1812"
  panel: "#221d16"
  panel-border: "#3a3226"
  document-paper: "#ddd0ac"
  document-paper-bright: "#e9dfc2"
  inspection-lamp: "#eaa73e"
  inspection-lamp-bright: "#f6c065"
  inspection-lamp-dim: "#8a5f22"
  verdict-flagged: "#c8412c"
  verdict-flagged-dim: "#4a1d14"
  verdict-cleared: "#2f8f68"
  verdict-cleared-dim: "#163829"
  text-primary: "#ede6d6"
  text-dim: "#a89d87"
  text-faint: "#6f6650"
typography:
  display:
    fontFamily: "Space Grotesk, Segoe UI, sans-serif"
    fontSize: "clamp(2.6rem, 6vw, 5rem)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.02em"
  body:
    fontFamily: "IBM Plex Sans, Segoe UI, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "0.72rem"
    fontWeight: 400
    letterSpacing: "0.08em"
rounded:
  sm: "3px"
  md: "4px"
  lg: "6px"
  xl: "8px"
spacing:
  sm: "0.6rem"
  md: "1.2rem"
  lg: "2.5rem"
  xl: "7rem"
components:
  button-primary:
    backgroundColor: "{colors.inspection-lamp}"
    textColor: "#241a08"
    rounded: "{rounded.md}"
    padding: "0.7rem 1.3rem"
  button-primary-hover:
    backgroundColor: "{colors.inspection-lamp-bright}"
  verdict-flagged:
    backgroundColor: "transparent"
    textColor: "{colors.verdict-flagged}"
    rounded: "{rounded.md}"
  verdict-cleared:
    backgroundColor: "transparent"
    textColor: "{colors.verdict-cleared}"
    rounded: "{rounded.md}"
---

# Design System: Phishing Email Inspection Desk

## Overview

**Creative North Star: "The Document Authentication Checkpoint"**

The system dramatizes the product's actual mechanism — a forensic inspection counter under
raking/UV light, examining a document for authenticity and stamping a verdict — rather than
illustrating it as a decorative metaphor. This was a rolled, weighed design direction (not a
default choice): assigned by a dice-roll process against a resonance-ordered list of visual
systems the audience would recognize, then weighed against catalog challengers (a particle-
collider event display, a demoscene/cracktro terminal world) on audience-identification and
product-clarity. Two disciplines were donated in from those declined-but-competitive
challengers: confidence renders as real visual weight (bar length, brightness, and glow all
scale off the same measured number, never an inert percentage label), and a persistent live
monospace readout register carries real-time state (the health badge).

This explicitly refuses two category defaults: the generic AI-dashboard hero-metric template
(big number, small label, accent), and the "hacker green-on-black" cybersecurity cliché.
Instead: a warm, dim case-file room at rest, lit by one dominant amber inspection lamp.

**Key Characteristics:**
- One dominant saturated accent (Committed color strategy) — the inspection-lamp amber
  covers real page area (the hero's light-beam gradient, primary actions, data highlights),
  not just a button tint.
- Two strict semantic colors (crimson/teal) reserved *only* for the phishing/safe verdict —
  never used decoratively elsewhere in the system.
- The predict tool is the first-viewport thesis, not a form under a marketing header.
- No kicker/eyebrow labels anywhere — headings carry their own framing.
- One authored motion signature (the hero's raking-light scan → rubber-stamp verdict);
  every other section either drops scroll motion or uses a materially different mechanism,
  so nothing else competes with it.

## Colors

A warm, dim case-file room at rest — near-black but never cold — lit by one dominant amber
inspection lamp; verdict colors are load-bearing signals, never decoration.

### Primary
- **Inspection Lamp** (`#eaa73e`, bright variant `#f6c065`, dim variant `#8a5f22`): the one
  dominant saturated color (Committed strategy — covers 30-60% of the hero via the light-beam
  gradient). Used for primary actions, the hero's accent word, data highlights, and focus
  states. Never used for a verdict.

### Secondary — verdict semantics (strict, never decorative)
- **Verdict Flagged** (`#c8412c`, dim `#4a1d14`): the phishing verdict only — the stamp
  border/text, the confidence gauge fill, the ledger's danger state if ever needed.
- **Verdict Cleared** (`#2f8f68`, dim `#163829`): the safe verdict only, and the deployment
  certificate's "verified" register (a distinct but related use: proof that held up).

### Neutral
- **Case-File Ink** (`#15120e`) / **Ink-2** (`#1c1812`): page and alternating-section
  backgrounds.
- **Panel** (`#221d16`) with **Panel Border** (`#3a3226`): cards, tables, the verdict readout
  panel.
- **Document Paper** (`#ddd0ac`, bright `#e9dfc2`): reserved *only* for the literal
  document-under-inspection surface (the textarea panel) — never used as a page background,
  specifically to avoid the generic warm-cream-paper AI-design default.
- **Text Primary** (`#ede6d6`) / **Text Dim** (`#a89d87`) / **Text Faint** (`#6f6650`): a
  three-step warm-neutral text ramp for hierarchy on the dark ground.

### Named Rules
**The One Lamp Rule.** Amber is the only color allowed to carry real page area outside the
document panel and the two verdict colors. If a new element needs emphasis, it borrows amber
or restrained neutral weight — never a new hue.

**The Verdict-Only Rule.** Crimson and teal render a phishing/safe verdict and nothing else.
A status chip, a table highlight, an icon — none of it borrows verdict color for unrelated
"success/error" UI states; introduce a different mechanism (weight, position, the amber
accent) instead.

## Typography

**Display Font:** Space Grotesk (with Segoe UI, sans-serif fallback)
**Body Font:** IBM Plex Sans (with Segoe UI, sans-serif fallback)
**Label/Mono Font:** IBM Plex Mono (with ui-monospace, monospace fallback)

**Character:** Space Grotesk's geometric, slightly technical character reads as instrument-
panel/official-documentation rather than a generic startup sans; IBM Plex Mono is reserved
for genuine data and measurement (never as a "technical" costume), tying the ledger's
numbers, the health badge's live status, and the terminal-style curl transcript into one
register.

### Hierarchy
- **Display** (700, `clamp(2.6rem, 6vw, 5rem)`, 1.05 line-height, -0.02em tracking): the hero
  title and each section's h2 — the thesis line, not a generic header treatment.
- **Body** (400, 1.15rem hero / 0.98rem section body, 1.6 line-height, measure capped at
  68ch): all prose.
- **Label/Data** (400-500, 0.68-0.82rem, 0.06-0.12em tracking, mono, often uppercase): live
  status, table headers, case data, confidence readouts — always real data or measurement,
  never decorative.

### Named Rules
**The Mono-Means-Real Rule.** Monospace type appears only where the content is a genuine
measured number, timestamp, status string, or code/URL — never as a "technical" visual
costume on prose.

## Layout

Single-column sections at `max-width: 1180px`, generous vertical rhythm (`padding: 7rem 6vw`
per section, alternating `--ink`/`--ink-2` backgrounds to separate them without borders). The
hero is the exception: full `100svh`, its own two-column "desk" grid (document panel +
verdict readout) that collapses to one column under 860px. Body copy is capped at a 68ch
measure throughout. Mobile: the health badge and any two-item header row stacks vertically
under 560px rather than wrapping awkwardly; data tables scroll horizontally inside their own
container rather than compressing columns unreadably.

## Elevation & Depth

Mostly flat, tonal layering (panel vs. ink vs. ink-2) does the separation work, not shadows.
The one deliberate exception is the document panel and the verdict stamp: the document panel
carries a real offset+blur shadow (`0 30px 60px -20px rgba(0,0,0,0.55)`) to read as a
physical object lifted off the desk, and the confidence gauge's fill glows
(`box-shadow` scaled to the live confidence value) as an instrument-panel readout, not
decoration.

### Named Rules
**The Physical-Object Rule.** A real depth shadow is earned only by an element the direction
treats as a physical object in the scene (the document panel) or an active instrument
readout (the confidence gauge). Everything else stays flat tonal layering.

## Shapes

Small, consistent radii (3-8px) throughout — never fully rounded/pill shapes except status
chips, which use `border-radius: 3px` (not `999px`) to stay in the same technical-document
register as everything else. The verdict stamp is the one deliberate exception to
rectilinear geometry: a `-6deg` rotation and a 3px solid border read as a physical rubber
stamp impression, not a UI card.

## Components

### Buttons
- **Shape:** 4px radius, no pill shapes.
- **Primary ("Inspect"):** amber background (`#eaa73e`), near-black text (`#241a08`),
  `0.7rem 1.3rem` padding, 500 weight, 0.06em tracking.
- **Hover:** brightens to `#f6c065` and lifts 1px (`translateY(-1px)`).
- **Secondary ("load sample"):** ghost — transparent background, 1px border in the
  document panel's ink tone at 33% opacity, tightens to full opacity on hover.

### Cards / Panels
- **Corner style:** 6-8px radius.
- **Background:** `--panel` (`#221d16`) on the dark ground; `--paper` (`#ddd0ac`) for the
  one physical-document exception.
- **Border:** 1px `--panel-border` (`#3a3226`), or the verdict-cleared dim tone for the
  deployment certificate specifically (a panel that itself represents a "passed" state).
- **Shadow strategy:** flat by default; only the document panel and verdict stamp carry
  real depth (see Elevation & Depth).

### Inputs / Fields
- **Style:** the document textarea sits on `--paper-2`, 1px `#b8a878` border, 4px radius —
  deliberately styled as paper, not a standard dark-mode input.
- **Focus:** 2px outline in the lamp-dim tone, 1px offset.

### The Verdict Stamp (signature component)
A rubber-stamp-style badge (3px solid border in the verdict color, `-6deg` rotation, spring
entrance animation) that lands after the raking-light scan completes. This is the product's
single most distinctive visual moment — it must never be replaced with a generic toast,
badge, or checkmark icon.

### The Confidence Gauge (signature component)
A horizontal instrument bar whose fill width, brightness, and glow all scale directly off
the model's real confidence number — the donated discipline from the particle-collider
challenger. Never render a confidence value as a bare percentage without this gauge.

### Navigation
No persistent nav bar — this is a single long-scroll page. The only persistent chrome is the
health badge (top-right of the hero, live-polling `/healthz` every 15s), which functions as
the system's one "you are here / it's working" signal.

## Do's and Don'ts

### Do:
- **Do** reserve crimson/teal strictly for the phishing/safe verdict (The Verdict-Only Rule).
- **Do** let the amber inspection-lamp accent cover real page area, not just button tints
  (The One Lamp Rule).
- **Do** back every number shown (accuracy, latency, memory, confidence) with a real
  measured value — this is a stated product principle (PRODUCT.md), not just a design
  preference.
- **Do** give each new section its own distinct motion mechanism if it needs one at all; the
  hero's scan-to-stamp is the system's one signature moment.

### Don't:
- **Don't** add a kicker/eyebrow label above any heading — this was a material finding in
  this project's own finish review and is a hard ban, not a style preference.
- **Don't** reuse the generic `whileInView` fade-up-on-scroll recipe on more than one
  section; it was found and removed as duplicated boilerplate.
- **Don't** use the document-paper tone as a page background — it's reserved for the literal
  inspected-document surface only.
- **Don't** let any UI copy or interaction read as *generating* phishing content, even
  illustratively — detection-only is a hard product constraint (PRODUCT.md).
