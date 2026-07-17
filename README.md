# Instrument Tagger

A dockerfile that, when started, runs a script to scan media files using 
Essentia and to label them either as INSTRUMENTAL or VOCAL.  

This allows for other software (e.g. LMS/Lyrion Dynamic Playlist Creator)
to generate playlists of "instrumental" music; great for focusing.

# Status

Basic working container. Agentic coded. Will human review and remove any
junk ASAP.

# TODO

Planning to add:
- Option to tag all songs with "INSTRUMENTAL_CONFIDENCE" tag - preserving 
  analysis results
- Option to re-tag all songs using the confidence tag but with a different
  user-decided threshold for what accounts for "instrumental" (default 50%)
- Option to skip re-analysis for tracks that already have a confidence tag.
