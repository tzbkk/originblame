# OriginBlame — CIKM 2026 Demo Video Script (3 minutes)

Format: Slides + webapp screen recording, English TTS narration (Chatterbox).

---

## [Slide 1] The Provenance Gap (0:00–0:30)

**Visual**: Three HuggingFace dataset screenshots in sequence:
1. Dataset overview — rich metadata (language, license, task) but no contributor provenance
2. Second dataset — same gap
3. Third dataset — a record with `contributor: ameya-2003` (Ethereum smart contract prompt). Pause here.
   Arrow → pipeline diagram: tokenization → packing → "?" — contributor identity lost after processing.

**Narration** (~70 words):

> Across HuggingFace, training datasets carry rich metadata—languages, licenses, task types—but no standardized contributor provenance. Some datasets do record contributors at collection time. Here, ameya-2003 contributed an Ethereum smart contract prompt. But once this text enters a training pipeline—tokenization, packing, deduplication—that contributor link is severed. When someone requests removal, trainers have no way to trace which training records came from them, forcing massive over-deletion across the entire dataset.

---

## [Slide 2] OriginBlame (0:30–1:00)

**Visual**: Figure 2 — three-layer architecture diagram (Authors → Sections → Document-Index with hash sharding).

**Narration** (~75 words):

> OriginBlame fills this gap with a three-tier content-addressable architecture. The Authors layer stores contributor identities. The Sections layer registers copyright metadata—each section identified by a content-addressable hash of source path, authors, contributors, license, and year. The Document-Index layer links every output record to its contributing sections via a line hash of record content and a sources list of section hashes—all sharded into 256 buckets for sub-millisecond provenance queries. Three-level revocation—author, section, record—is fully reversible, constant-time. No database, no GPU, fully deterministic.

---

## [Screen Recording] Dataset Overview (0:45–1:10)

**Actions**: Open webapp → Dataset Overview page → switch zhwiki 1k → ChatML 36k → kernel.

**Narration** (~55 words):

> The web application serves eight indexed datasets across three domains: Chinese Wikipedia at four scales, Linux kernel source with git-blame attribution, and ChatML training data. Switching datasets reveals the author-versus-contributor distinction—authors are the editors whose content is currently visible, while contributors are historical editors whose changes were later overwritten.

---

## [Screen Recording] Provenance Query (1:10–1:40)

**Actions**: Author Browser → search "InternetArchiveBot" → click detail (3,618 sections, 12,981 as author / 30,979 as contributor) → click into a record → show full provenance chain (doc hash → section hash → authors → contributors → license).

**Narration** (~50 words):

> Searching for InternetArchiveBot reveals over three thousand sections and thirteen thousand records as author, versus thirty thousand as contributor—each linking back to original Wikipedia editors. Any record resolves to its full provenance chain in under four milliseconds.

---

## [Screen Recording] Right-to-Erasure (1:40–2:25)

**Actions**: Navigate to Right-to-Erasure page → select author "Ohtashinichiro" → impact preview loads → show comparison chart (dataset-level 9,951 / contributor reference 1,191 / OriginBlame 541) → click "Execute Revocation" → records marked revoked.

**Narration** (~80 words):

> The Right-to-Erasure page demonstrates three-level revocation. Selecting author Ohtashinichiro triggers an impact preview: dataset-level deletion would destroy nearly ten thousand sections. Even tracking contributors—which ob does—would still delete over a thousand. OriginBlame distinguishes registered authors from contributors, targeting only five hundred forty-one—an eighteen-fold reduction in over-deletion. Executing the revocation instantly marks all affected sections.

---

## [Screen Recording] Undo + Audit (2:25–2:45)

**Actions**: Click "Undo Revocation" → records restored → navigate to Audit Log → show timestamped entries (register, revoke, restore).

**Narration** (~35 words):

> Revocation is fully reversible. The audit trail logs every operation—registration, revocation, restoration—providing compliance evidence for GDPR Article seventeen.

---

## [Slide 3] Availability (2:45–3:00)

**Visual**: White slide with links:
- github.com/tzbkk/originblame (MIT license)
- github.com/tzbkk/rust-originblame (MIT license)
- Live demo: [URL]
- Paper: [DOI]

**Narration** (~25 words):

> Open source under MIT license. Try the live demo or explore the code at these URLs.

---

## Production Checklist

- [ ] Capture HF datasets page screenshots (3 screenshots with annotations)
- [ ] Make Slide 1 (HF problem) — Keynote/Google Slides
- [ ] Make Slide 2 (architecture) — reuse paper Figure 2
- [ ] Make Slide 3 (availability) — URLs + QR code
- [ ] Record webapp screen (OBS, 1920×1080, 30fps)
- [ ] Generate TTS narration (Chatterbox, `exaggeration=0.3, cfg_weight=0.3`)
- [ ] Assemble in DaVinci Resolve / Premiere / Kdenlive
- [ ] Export 1080p H.264, upload to YouTube unlisted
- [ ] Replace `[PENDING]` in demo.tex with YouTube URL
