#!/usr/bin/env python3
"""Generate demo video narration using Chatterbox TTS.

Generates each segment as a single TTS call for consistent pacing,
with trailing silence padding for breathing room.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    python generate_narration.py [--voice REF.wav]

Outputs WAV files to narration/ directory.
"""

import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

EXAG = 0.1
CFG = 0.1
TAIL_SILENCE = 1.5  # seconds of silence at end of each segment

SEGMENTS = [
    (
        "01_problem.wav",
        "Across HuggingFace, training datasets carry rich metadata—"
        "languages, licenses, task types—but no standardized contributor provenance. "
        "Some datasets do record contributors at collection time. "
        "Here, ameya-2003 contributed an Ethereum smart contract prompt. "
        "But once this text enters a training pipeline—"
        "tokenization, packing, deduplication—"
        "that contributor link is severed. "
        "When someone requests removal, "
        "trainers have no way to trace which training records came from them, "
        "forcing massive over-deletion across the entire dataset.",
    ),
    (
        "02_originblame.wav",
        "OriginBlame fills this gap with a three-tier content-addressable architecture. "
        "The Authors layer stores contributor identities. "
        "The Sections layer registers copyright metadata—"
        "each section identified by a content-addressable hash of "
        "source path, authors, contributors, license, and year. "
        "The Document-Index layer links every output record to its contributing sections "
        "via a line hash of record content and a sources list of section hashes—"
        "all sharded into 256 buckets for sub-millisecond provenance queries. "
        "Three-level revocation—author, section, record—"
        "is fully reversible, constant-time. "
        "No database, no GPU, fully deterministic.",
    ),
    (
        "03_overview.wav",
        "The web application serves eight indexed datasets across three domains: "
        "Chinese Wikipedia at four scales, "
        "Linux kernel source with git-blame attribution, "
        "and ChatML training data. "
        "Switching datasets reveals the author-versus-contributor distinction—"
        "authors are the editors whose content is currently visible, "
        "while contributors are historical editors whose changes were later overwritten.",
    ),
    (
        "04_provenance.wav",
        "Searching for InternetArchiveBot reveals over three thousand sections "
        "and thirteen thousand records as author, "
        "versus thirty thousand as contributor—"
        "each linking back to original Wikipedia editors. "
        "Any record resolves to its full provenance chain in under four milliseconds.",
    ),
    (
        "05_erasure.wav",
        "The Right-to-Erasure page demonstrates three-level revocation. "
        "Selecting author Ohtashinichiro triggers an impact preview: "
        "dataset-level deletion would destroy nearly ten thousand sections. "
        "Even tracking contributors—which ob does—"
        "would still delete over a thousand. "
        "OriginBlame distinguishes registered authors from contributors, "
        "targeting only five hundred forty-one—"
        "an eighteen-fold reduction in over-deletion. "
        "Executing the revocation instantly marks all affected sections.",
    ),
    (
        "06_undo_audit.wav",
        "Revocation is fully reversible. "
        "The audit trail logs every operation—"
        "registration, revocation, restoration—"
        "providing compliance evidence for GDPR Article seventeen.",
    ),
    (
        "07_availability.wav",
        "Open source under MIT license. "
        "Try the live demo or explore the code at these URLs.",
    ),
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "narration")


def main():
    voice_ref = None
    if "--voice" in sys.argv:
        idx = sys.argv.index("--voice")
        voice_ref = sys.argv[idx + 1]
        if not os.path.isfile(voice_ref):
            print(f"Error: voice reference not found: {voice_ref}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Chatterbox model...")
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    sr = model.sr
    tail = torch.zeros(1, int(sr * TAIL_SILENCE))
    print(f"Model loaded in {time.time() - t0:.1f}s  (sr={sr}, tail_silence={TAIL_SILENCE}s)")

    for i, (filename, text) in enumerate(SEGMENTS):
        out_path = os.path.join(OUTPUT_DIR, filename)
        print(f"\n[{i+1}/{len(SEGMENTS)}] {filename}")

        t0 = time.time()
        wav = model.generate(text, audio_prompt_path=voice_ref,
                             exaggeration=EXAG, cfg_weight=CFG)
        speech_dur = wav.shape[-1] / sr
        print(f"  speech: {speech_dur:.1f}s  (gen: {time.time() - t0:.1f}s)")

        padded = torch.cat([wav, tail], dim=-1)
        total_dur = padded.shape[-1] / sr
        ta.save(out_path, padded, sr)
        print(f"  saved: {out_path}  (total: {total_dur:.1f}s)")

    print(f"\nAll {len(SEGMENTS)} segments saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
