# Instrument Tagger

A dockerfile that, when started, runs a script to scan media files using 
Essentia and to label them either as INSTRUMENTAL or VOCAL.  

This allows for other software (e.g. LMS/Lyrion Dynamic Playlist Creator)
to generate playlists of "instrumental" music; great for focusing.

# Status

Basic working container. Agentic coded. Will human review and remove any
junk ASAP.

# Configuration

All options are set via environment variables:

- `INSTRUMENTAL_THRESHOLD` (default `50`) — confidence percentage (0-100) at or
  above which a track is labeled INSTRUMENTAL. Change it and re-run to re-label
  the whole library from stored confidence, without re-analyzing.
- `STORE_CONFIDENCE` (default `true`) — writes the raw ML confidence to an
  `INSTRUMENTAL_CONFIDENCE` tag (0-100%), preserving analysis results.
- `SKIP_ANALYZED` (default `true`) — tracks that already have a confidence tag
  are re-labeled from that value instead of being re-analyzed. Set to `false`
  to force fresh ML analysis of everything.
- `FORCE_RETAG` (default `false`) — overwrites existing tags with fresh ML
  predictions, ignoring stored confidence.
