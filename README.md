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

## DJ Mode (Ctrl+D)

The decks dock above the library, Traktor-style. You can pull them out
into their own window.

- **Loading tracks:**
  - Right-click any track (library, Next Track suggestions or setlist) and
    choose **Load to Deck A/B/C/D**. A playing deck is greyed out, so a
    track that is on air is never replaced.
  - Drag a track onto a deck.
  - Keyboard: **Ctrl+←/→** loads the selected track to deck A/B;
    **Ctrl+Shift+←/→** loads it to deck C/D.
- **Centre mixer:** each channel has gain, EQ high/mid/low (kill when fully
  left), a **filter** knob (left = low-pass, right = high-pass), FX 1/2
  assign, headphone cue, a level meter and the channel fader. Below them:
  crossfader, master volume and meter, headphone cue mix and volume, and
  **REC** (records the mix to a 24-bit WAV file).
- **Decks:**
  - Waveforms coloured by frequency (red = bass, green = mids, blue =
    highs), with a beat grid and a scrolling zoom (mouse wheel).
  - 8 **hot cues**, saved per track.
  - Loop IN/OUT, ½ and ×2, auto loops from 1 to 16 beats, **beat jump**.
  - **Flux** mode: after a loop or jump, the track carries on where it
    would have been.
  - **Key shift** of ±12 semitones, plus **KEY SYNC**: the smallest
    transposition that makes the deck harmonic with the master.
  - Beat-grid nudge and "set beat 1 here", saved per track.
- **Cue point and mix markers:**
  - When a track loads, its cue point is set to the **first real beat** (the
    first kick), skipping silence and beatless intros.
  - Right-click the waveform to choose **Set mix-in (intro) here**, **Set
    mix-out (outro) here** or **Set cue point here**. Each snaps to the
    start of the bar you clicked and is saved with the track.
  - The IN and OUT flags appear on the waveform, and the shaded area after
    OUT shows the blend.
- **Beat 1 is found from the music itself.** In dance music, phrase changes
  (the kick drops out or comes back, a breakdown starts, a drop hits)
  happen on the downbeat. CTA uses those moments to fix each track's beat
  grid and phrase start, so tracks are blended bar-for-bar on the right
  beat. This runs once per track and the result is saved. Grids you have
  edited by hand are never touched.
- **Trance and progressive** are recognised from the AI style and get
  16-bar phrasing and long blends.
- **Learns from your mixes:** every transition you do by hand is recorded:
  where in each track's structure you mixed, how long the blend was, and
  how far before the next track's drop you brought it in. After 3
  examples for a style, auto-mix mixes that style your way. The auto-mix
  panel shows how much it has learned and has a **Forget** button.
- **Never the same song twice:** auto-mix and "auto-load best next track"
  skip anything played this session, queued, loading or on a deck,
  including other files of the same song.
- **Played column** in the library: `● now` means played this session,
  `✓ 3×` means played before. Hover to see the last time. Tracks already
  played this session are dimmed.
- **Scratching:** drag the scrolling waveform, or grab the spinning jog
  platter. When you let go, the track carries on. On a paused deck,
  dragging searches through the track. Double-click the waveform to jump
  to that point.
- **How auto-mix DJs:**
  - **It reads each track's structure.** Every 8-bar phrase is labelled
    intro, groove, build, drop, breakdown or outro, from the kick, bass,
    mid and high energy on each bar.
  - **It mixes out in the outro**, right after the last drop has fully
    played. It never cuts a drop or a breakdown mid-phrase.
  - **It mixes in from the next track's intro**, timed so that track's
    **drop lands exactly when the blend ends**. The **bass swap happens on
    that downbeat**: the incoming bass stays cut until its drop, then the
    two basses swap on the "one".
  - **It loops to keep the beat.** If the outgoing track has no outro, or
    one that's too short, CTA loops its last solid 4–8 bars of groove
    until the swap.
  - **It echoes out on a clash.** When the keys don't mix or the tempo gap
    is too wide, the outgoing track gets a beat-synced echo tail and the
    next track drops in clean on the downbeat.
  - **It mixes with EQ, not just volume.** The incoming track comes in highs
    first with mids reduced; the outgoing track's highs and mids tail off
    after the swap.
  - **Blend length depends on genre:** long blends for house and techno,
    short ones for hip-hop and pop. Your bars setting is the maximum.
  - **A peak limiter guards the master**, so two loud tracks overlapping
    never distort.
- **Auto-mix timing:**
  - Choose the blend length with the 4 / 8 / 16 / 32-bar buttons, or type
    any value from 1 to 64.
  - Auto-mix starts the blend at the outgoing track's outro marker and
    brings the next track in at its intro marker.
  - Without markers, it ends the blend on the last bar that still has
    beats and starts the next track on its first beat, so there is no dead
    air.
- **2 beat-synced FX units:** Delay, Reverb, Flanger, Gater, Crush, each
  with dry/wet, amount and beat timing. The units are inserts that you
  assign per channel.
- **📡 Stream to phone:** CTA serves the live mix on your network. Open the
  address it shows (e.g. `http://192.168.1.20:8000`) in your phone's
  browser to listen and see what is playing. It works on the same Wi-Fi;
  away from home, use a VPN such as Tailscale or forward the port. Windows
  will ask once to allow CTA through the firewall.
- **History:** every track that plays for 30 s or more is logged. You can
  export the list as M3U8, **Traktor NML**, rekordbox XML or CSV.
- Setlists also export as **Traktor NML** (with BPM and key).


Under the hood:

- **Key lock**: change the tempo without changing the key. It uses Rubber
  Band R3 real-time time-stretching, the same library Mixxx uses, at about
  3% of one CPU core per deck. Turn key lock off for vinyl-style pitch.
- **Beatmatching and phase matching**: every track gets a beat grid (the
  tempo plus where beat 1 falls), fitted when it loads and then cached.
  **SYNC** matches the tempo, including half/double time, and keeps the
  beats phase-locked to the master. The phase meter shows how close they
  are. Pressing play on a synced deck starts it exactly on the beat.
- **Auto-mix** plays your setlist, or picks the best next track itself, as a
  continuous mix:
  - The mix-out point is chosen from each track's energy profile and
    aligned to an 8-bar phrase.
  - The next track starts on its first downbeat, tempo-synced and
    phase-locked.
  - The blend is an equal-power fade with a bass swap halfway, so two
    basslines never play at full level together.
  - Afterwards the tempo glides to the new track's own BPM. Key lock keeps
    the pitch the same, so a set can move from 122 to 128 BPM without a
    jump.
- **Assist mode**, for when you want to DJ yourself:
  - SYNC and key lock.
  - Quantised play and cue.
  - **Auto transition now**: CTA performs the blend into the other deck
    from the next bar.
  - The best next track is loaded automatically when a deck is empty.
  - Touching a fader, EQ or the crossfader during an automatic blend hands
    control straight back to you.
- **Pioneer DDJ-T1**:
  - Plug it in and CTA finds both its MIDI ports and its 4-channel sound
    card: master on outputs 1-2, headphones on 3-4, with headphone cue and
    a cue/master mix.
  - A basic map (play, cue, jog, tempo, load, browse) works straight away.
    **Controller setup…** then walks you through every control once ("move
    the deck A tempo fader…"), so the mapping matches your unit exactly. It
    is saved, and LEDs follow the deck state.
  - Jog wheel: touch the top to scratch, turn the outer ring to nudge, or
    search when paused.
  - Any other MIDI controller can be mapped with the same wizard.
- Loops (1–32 beats), cue points snapped to beats, 3-band kill EQ with
  Linkwitz-Riley crossovers, trim, and auto-gain from each track's measured
  loudness.

Audio dropouts? Open **Audio…** and raise the buffer. 512 samples
(~12 ms) suits most PCs.

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

The bundled Discogs-EffNet model is licensed **CC BY-NC-SA 4.0**
(non-commercial use) by MTG/UPF. DJ Mode's key lock uses **Rubber Band
(GPL v2+)**, so the distributed builds are GPL-licensed. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
