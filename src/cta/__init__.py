"""Tempest - DJ library analyser: BPM, key, AI genre and instant setlists."""

VERSION = "2.3.5"             # the release the DJ names (major.minor.patch)
try:                          # CI writes the build number into _build.py; a source checkout has none
    from ._build import BUILD
except ImportError:
    BUILD = 0
# every build has its own number: 2.3.5.142 - fixes ship as new builds, not new versions
__version__ = f"{VERSION}.{BUILD}" if BUILD else VERSION
APP_NAME = "Tempest"          # the app, the bundle, the data folder
LEGACY_NAME = "CTA"           # what it was called up to 2.2.x (data, settings and installs migrate)
