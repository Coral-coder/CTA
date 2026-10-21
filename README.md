# CTA: DJ library analyser and instant setlist builder

CTA scans your music library and works out the **BPM, key (with Camelot
code), energy and genre/style** of every track. From any seed track it then
suggests **what to mix next**, or builds a whole **setlist** in milliseconds.

- **Fast scans.** Analysis runs in parallel across every CPU core. The AI
  model runs in batches on the GPU (DirectML on Windows, so any DX12 GPU
  works: NVIDIA, AMD or Intel; CoreML on macOS). Re-scans only look at new
  or changed files.
- **Instant queries.** The whole library is held in memory as vectors, so
  next-track suggestions take about 1 ms and a 40-track setlist from a
  50k-track library takes well under a second.
- **Built to stay up.** One corrupt file can't crash a scan: decoder crashes
  are isolated in worker processes, retried, and then recorded as failed.
  Data lives in SQLite (WAL mode), and you can stop and resume a scan at any
  time.

## Download

Get the latest build from the
[Releases](https://github.com/coral-coder/CTA/releases) page.

| Platform | File | Run |
|---|---|---|
| Windows 10/11 x64 | `CTA-x.y.z-windows-x64.zip` | unzip, then run `CTA\CTA.exe` |
| macOS (Apple Silicon) | `CTA-x.y.z-macos-arm64.zip` | unzip, then right-click `CTA.app` and choose **Open** (needed the first time only) |

The AI model is bundled, so nothing needs to be installed.

## Using it

1. **Add Folder…** and pick your music folder. The first scan starts on its
   own. A progress bar shows files per second and the time remaining.
2. Click any track. The **Next Track** tab ranks every compatible track by:
   - **Tempo**: how far you'd have to move the pitch fader, within your
     ±% range. Half and double time count (87 ↔ 174).
   - **Key**: Camelot wheel rules: same key, ±1, relative major/minor,
     diagonal mixes and energy-boost moves.
   - **Sound**: similarity of the AI audio embeddings ("sounds like").
   - **Genre**: overlap between the AI style profiles of the two tracks.
   - **Energy**: hold the level, lift it or drop it.

   Weights and limits are adjustable. Double-click a suggestion to
   **follow** it and see what mixes out of that one, so you can walk a set
   live.
3. **Setlist Builder**: choose the number of tracks, an optional tempo ramp
   (for example 122 → 128 BPM), an energy curve (steady / build / peak /
   wave / cooldown) and a variety amount. Then click **Generate**. A beam
   search finds the sequence with the best transitions overall. Each
   transition shows its pitch change and mix score.
4. **Export** to:
   - **M3U8**: Serato, Traktor, VirtualDJ, Engine DJ and rekordbox
   - **rekordbox XML**: includes BPM and key
   - **CSV**

Right-click a track to double or halve its BPM, or to set the key by hand.
Your corrections are kept across re-scans.

## Should AI be used? (16 GB GPU)

AI is used where it beats classic signal processing:

| Task | Method | Why |
|---|---|---|
| BPM | DSP: onset autocorrelation plus a harmonic DFT refined to 0.005 BPM | Accurate and fast for a beat grid. A neural net gains little here. |
| Key | DSP: HPCP chroma with tuning correction, Shaath/KeyFinder profiles | The standard approach used in DJ software |
| Genre / style | **AI**: Discogs-EffNet, 400 Discogs styles | Tags are often missing or wrong. The model knows Tech House from Deep House. |
| "Sounds like" | **AI**: 1280-d EffNet embeddings | Lets the next track match the vibe, not only BPM and key |
| Half/double-time fixes | **AI style** + rhythm analysis | For example, Drum & Bass is put at 174, not 87 |

The model is about 18 MB and uses well under 1 GB of VRAM, far inside a 16 GB
card. A GPU speeds up the AI step, but decoding and DSP run on the CPU cores.
With 128 GB of RAM, CTA can run one worker process per core (each needs a
few hundred MB at most). Expect about 1–3 tracks per second per CPU core.
A 16-core desktop gets through a 10,000-track library in roughly 5–10 minutes.
Later re-scans take seconds.

The model input was checked against Essentia's reference implementation (it
matches to within 1e-3). There is a unit test for this.

## Updates

CTA keeps itself up to date. At start-up (at most every 12 hours) it checks
GitHub Releases. When a new version is out, you can **Update now**, **Later**
or **Skip this version**. You can also click **Check for Updates** in the
toolbar at any time, or turn automatic checks off in Settings.

An update goes through these steps:

1. The zip for your platform is downloaded and checked against the
   release's SHA-256 checksums.
2. It is unpacked next to the install and test-run once.
3. Only then does CTA close. A small helper swaps the folders, keeps the old
   version until the swap has worked (and rolls back if it fails), and
   restarts CTA.

Your library, settings and analysis data live in the user data folder and
are never touched. CTA must sit in a folder you can write to, such as
Desktop, Documents or `C:\Tools`. In Program Files it offers the download
page instead.

From the command line: `cta update --check` or `cta update`.

## Command line

The Windows build includes `cta-cli.exe`. On macOS, run
`CTA.app/Contents/MacOS/CTA <command>`.

```
cta scan "D:\Music"                     # add a folder and analyse it
cta list --search "tech house"
cta next "artist - title"               # best next tracks
cta setlist "artist - title" -n 30 --end-bpm 128 --curve build -o set.m3u8
cta selftest                            # check that decoding, DSP, AI and DB all work
```

## Development

```
pip install -e ".[dev]"
pytest -q
python -m cta            # start the GUI
pyinstaller packaging/cta.spec
```

CI (`.github/workflows/ci.yml`) runs lint and tests on Windows, macOS and
Linux. It then builds the Windows and macOS executables and runs the
self-test and a GUI smoke test against the **frozen** build. Pushing a `v*`
tag publishes a GitHub release with both zips.

Your data lives in `%LOCALAPPDATA%\CTA` (Windows) or
`~/Library/Application Support/CTA` (macOS). The log is in `cta.log` in the
same folder.

## Licences

CTA's code is yours. The bundled Discogs-EffNet model is licensed
**CC BY-NC-SA 4.0** (non-commercial use) by MTG/UPF. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
