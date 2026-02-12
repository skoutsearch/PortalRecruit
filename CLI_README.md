# CLI Usage (PortalRecruit)

## Commands

### Search
```
/home/jch903/.venv_310/bin/python cli.py search "unselfish clutch point guards"
```

### Search + Debug Metadata
```
/home/jch903/.venv_310/bin/python cli.py search "unselfish clutch point guards" --debug
```
Prints the raw Chroma metadata JSON for the top result to help diagnose available fields.

### Interactive Mode
```
/home/jch903/.venv_310/bin/python cli.py interactive
```
Type a query and press Enter. Type `q` or `exit` to quit.

## Current Features
- **Interactive mode** for repeated searches
- **--debug flag** for raw metadata inspection
- **Mock video links** when real video metadata is missing:
  `https://mock.synergy.com/video/<play_id>.mp4`

## Notes / TODO
- Replace Chroma local search with Synergy/SportRadar API calls when available.
- Replace mock video links with real Synergy/SportRadar video URLs.
