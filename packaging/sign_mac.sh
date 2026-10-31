#!/bin/bash
# Sign a PyInstaller .app for distribution outside the App Store (Developer ID +
# hardened runtime), inside out: every Mach-O file first, then the bundle.
#   sign_mac.sh <Tempest.app> <signing identity> [entitlements.plist]
set -euo pipefail
APP="$1"; ID="$2"; ENT="${3:-$(dirname "$0")/entitlements.plist}"
opts=(--force --options runtime --timestamp --sign "$ID")
# nested code (Python extension modules, dylibs, helper executables, frameworks)
find "$APP/Contents" -type f \( -name '*.dylib' -o -name '*.so' -o -perm -u+x \) -print0 | while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q 'Mach-O'; then
    codesign "${opts[@]}" --entitlements "$ENT" "$f"
  fi
done
find "$APP/Contents" -type d -name '*.framework' -print0 | while IFS= read -r -d '' f; do
  codesign "${opts[@]}" "$f"
done
codesign "${opts[@]}" --entitlements "$ENT" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
codesign -dvv "$APP" 2>&1 | grep -E 'Authority|TeamIdentifier|Timestamp' || true
