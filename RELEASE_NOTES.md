# Release Notes — v2.0 (The Century Update)

## Highlights
- Biometric search signals now surface alongside semantic ranking.
- Learned calibration powers the AI Position Engine with real weights.
- AI Vision analysis panel with parsed build/conditioning badges.
- Real-time search sliders for semantic (alpha) and size (beta) tuning.

# Release Notes — v1.0

## Highlights
- Semantic search stabilized with hybrid scoring, caching, and safer fallbacks.
- Evidence-based scout breakdowns now cite specific film actions when available.
- CLI tooling: interactive mode, debug metadata, mock video links, and scout breakdown output.
- Web UI: mock video links surface when real URLs are unavailable.
- AI Position Engine: learned calibration weights now drive position scoring (alpha=1.211, beta=3.074).

## Key Changes
- Normalized scoring to 0–1 for UI consistency.
- Added action-aware weighting (positive vs. negative play outcomes).
- Enriched LLM prompts with film notes and position/year context.
- Added player ID mapping to recover joins between plays and players.
- Backfilled tags for plays to improve semantic filtering.
- Added full-dataset calibration pipeline and model bundle loading across semantic + UI.

## Next
- Replace mock video links with Synergy/SportRadar URLs.
- Wire semantic search to production API endpoints.
- Expand metadata fields for deeper trait modeling.
