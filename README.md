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
4. **Playlists** (tab): make your own playlists. Drag tracks in from the
   library, or right-click a track → **Add to playlist**. Between every two
   songs CTA shows a **dimmed suggestion**: the library track that best
   bridges the gap (tempo, key, sound and genre into the next song).
   Double-click it to keep it, or right-click → **Suggest another bridge**.
   Drag to reorder; Delete removes. **Queue in auto-mix** plays the
   playlist as a mix; **To setlist** and **Export…** work as for setlists.
5. **Export** to:
   - **M3U8**: Serato, Traktor, VirtualDJ, Engine DJ and rekordbox
   - **rekordbox XML**: includes BPM and key
   - **CSV**

Right-click a track to double or halve its BPM, or to set the key by hand.
Your corrections are kept across re-scans.

## DJ Mode (Ctrl+D)

The decks sit above the library in a compact, Traktor-style layout. Drag
the bar between them to give the decks more or less room.

- **Global row (top):** FX 1, the auto-mix clock (master BPM, Auto-mix,
  skip, pause, blend length), the MAIN level and limiter, FX 2 and the
  recorder. Hide it under **Options**. The auto-mix queue is a tab next to
  the library.
- **Decks:** header (title, key, BPM, time), waveform with the tempo
  fader, SYNC / MASTER / KEY with beat jump, overview, then one transport
  row (play, CUE, CUP, FLX, loop sizes 1–32, IN / OUT / LOOP). The **⌃**
  button shows hot cues and the grid / key tools.
- **CUE** reacts the moment you press it. Hold it to preview from the cue
  point; drag off the button before letting go (or press PLAY while
  holding) and the track keeps playing. PLAY and CUP also act on press.
- **Jog platter:** the album art turns on the record.
- **Waveform colours (Options → Waveform colours):** **Rainbow** (default:
  the whole rainbow, red through orange, yellow, green, cyan and blue to
  violet, spread from each track's darkest to its brightest sound),
  **Spectrum** (bass red, mids green, highs blue), **Three-band**, **Fire**,
  **Ice** and **Classic**, with a **Saturation** slider (0–200 %).
- **Waveform zoom:** the **− / +** buttons under each tempo fader (the mouse
  wheel no longer zooms). **⛓ Sync zoom** keeps both decks at the same zoom.
  Double-click the overview to open the whole track, zoomable, with its
  markers.
- **Double-click resets** any knob, fader, the crossfader (centre) and the
  tempo fader.
- **Oscilloscope** next to the MAIN level shows the master output (triggered,
  auto-scaled). **Options → XY scope on the platters** draws each deck's
  stereo image (a goniometer: mono is a vertical line) over its record.
- **Search and filters** sit right above the track list.
- **⏱ MIX AHEAD (Time-Shift):** the button next to ▶ AUTO-MIX (also
  **Ctrl+T**, the auto-mix ▾ menu and the toolbar). Mix the next transition *now*, on your
  headphones, while the room keeps hearing the live decks; auto-mix plays
  your mix live later.
  - A separate preview player (not the live decks) with the outgoing and
    incoming tracks as zoomable full waveforms: place the IN / OUT markers
    precisely, click to scrub, ▶ to listen.
  - A preview mixer (faders, crossfader, EQ, filter). Tick **My controller
    mixes the preview** and your MIDI controller drives it instead of the
    live decks.
  - **● Rehearse:** the outgoing plays from 8 bars before its OUT marker,
    the incoming drops in on the OUT marker (or **Drop incoming now**) in
    phase. Mix it: every move is recorded on the beat until the outgoing is
    gone. **▶ Play back** lets you hear it as the room will.
  - When the live set reaches those two tracks, auto-mix starts the incoming
    on the same beat and replays your moves. Untick **Auto-mix plays this
    transition live** to keep it without using it.
- **Keyboard (Traktor Pro's default keys):** deck A on the top letter row,
  deck B on the home row. **Options → Keyboard shortcuts…** lists them;
  they never fire while you type in a text box.

  | | Deck A | Deck B |
  |---|---|---|
  | Cue (hold) | Q | A |
  | Play / pause | W | S |
  | Beat jump (Shift: back) | E | D |
  | Tempo bend (hold; Shift: slower) | T | G |
  | Sync (Shift: master) | U | J |
  | Tap BPM (Shift: snap phase) | I | K |
  | Loop in (Shift: halve) | O | L |
  | Loop out (Shift: double) | P | ; |
  | Loop on / off | [ | ' |
  | Key lock | Shift+Y | Shift+H |
  | Hot cues (Ctrl: delete; Shift: loop 1–16) | 1–5 | 6–0 |
  | Load selected track | Shift+← | Shift+→ |
- **Mixer:** GAIN, HI, MID, LOW, FILTER, FX and headphones per channel.
  **The gain, EQ, filter and key reset when a new track loads** (with a controller attached the knobs keep their place). The
  crossfader, headphone mix and volume sit under the decks.

- **Scalable decks:** each deck shows what fits its height, like Traktor's
  deck views:
  - **Full:** waveform, hot cues, loops, beat jump, key and grid tools,
    tempo fader and jog platter.
  - **Essential:** waveform, hot cues, loops, tempo and transport.
  - **Micro:** track header, overview waveform and transport.

  **View: Auto** switches by itself; you can also pin one view. The mixer
  sheds its GAIN and FILTER knobs, then the EQ, as it gets shorter. The jog
  platter can be switched off under **Options**; you can always scratch on
  the waveform.
- **Drag tracks into Traktor (or rekordbox, Serato, Finder / Explorer):**
  tracks dragged out of the library, the Next Track list or the setlist
  carry their audio files, so you can drop them on a Traktor deck or
  playlist. Select several (Shift / Ctrl / Cmd-click) to drag a batch, or
  a whole setlist.
- **Audio outputs (Audio… button):**
  - Devices are scanned **live, without stopping the music**: the list is
    re-read when you open it, and **Rescan** finds devices plugged in or
    paired while CTA runs.
  - **Extra master outputs:** add any number of extra devices (for example
    two or three Bluetooth speakers) under **Extra outputs**. Each plays
    the master mix, and CTA delays the faster ones so they all play in
    sync (Auto delay, or **Measure…** per speaker).
  - **Automatic detection:** CTA checks for devices every 10 seconds. A
    remembered speaker that comes back rejoins by itself; one that drops
    out is removed without touching the rest; a newly paired Bluetooth
    device shows an **Add … as output** button in the status bar.
  - **Clock (keeps time):** pick the output everything lines up with, or
    **Auto** (the slowest, usually a Bluetooth speaker). **Sync to the
    clock** chooses which outputs wait for it; untick your headphones to hear
    them with no added delay. **Clock adjust** (± ms, with −10 / −1 / +1 /
    +10 buttons) fine-tunes it by ear while the music plays; Cancel puts it
    back.
  - Master and headphones can each go to **any connected device**, for
    example the DDJ-T1's outputs, the laptop jack or a Bluetooth speaker.
    The channel pairs offered match each device.
  - **Bluetooth delay:** each output has a delay. **Auto** uses what the
    driver reports; for Bluetooth devices CTA suggests about 200 ms.
    **Measure…** gives you clicks on that output: tap along about 20 times
    and CTA works out how late it plays. With the delays known, CTA holds
    back the faster output so master and headphones stay in time, and
    draws the waveforms and playheads that late so what you see matches
    what you hear.
- **Small screens (13-inch laptops):** CTA fits a 1280-pixel-wide screen.
  For even more room, **Settings → Interface size** (70–150 %) scales the
  whole app after a restart.

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
  - Three markers, saved with the track and snapped to the start of a bar:
    **IN** (where the track is mixed in), **OUT** (where it starts to mix
    out) and **OUT END** (where it is gone, which fixes the blend length).
  - **✎ MARKERS** on each deck opens a zoomable full waveform with bar
    numbers. Wheel to zoom, drag to scroll, drag a flag to move it, or nudge
    it **−4 / +4 beats**. The **outro length** buttons (8, 16, 32, 64 or
    128 bars) set OUT END for you; past the end of the track, auto-mix loops
    the groove to last that long.
  - Right-clicking the deck waveform still sets IN, OUT, OUT END or the cue
    on that bar.
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
  how far before the next track's drop you brought it in. **Taking over
  an auto-mix transition** (any fader, EQ, crossfader, play, cue, tempo,
  loop or jog move on either deck, by mouse, keyboard or controller) makes
  auto-mix let go at once; the blend is learned as yours, and auto-mix
  carries on by itself a few seconds after your blend is done. After 3
  examples for a style, auto-mix mixes that style your way. The auto-mix
  panel shows how much it has learned and has a **Forget** button.
- **Sets stay in the seed's genre:** every suggestion is scored against the
  song the set started from (its AI style, sound and tempo), not just the
  previous track. A set started from electro house can't walk off into
  trance one "close enough" step at a time. When the closest tracks run
  out, it widens gradually. This applies to the phone's seed sets,
  auto-mix's own picks and the Setlist Builder.
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
  - **Long fills become part of the mix.** When the next track opens with
    a long beatless fill (pads, vocals, an ambient intro), the fill plays
    over the outgoing track's groove, looped if its outro is too short.
    The new beat lands as the old track leaves. When the outgoing track
    ends in a long beatless fill, the next track's beat comes in underneath
    it, with its intro looped until the fill is over. Fills are only
    layered when the keys match; otherwise they are skipped.
  - **It never hangs on dead air.** Before every blend it checks each bar
    for music and a beat, and moves or shortens the blend until there are
    no empty bars.
  - **It echoes out on a clash.** When the keys don't mix or the tempo gap
    is too wide, the outgoing track gets a beat-synced echo tail and the
    next track drops in clean on the downbeat.
  - **It mixes with EQ, not just volume.** The incoming track comes in highs
    first with mids reduced. The outgoing track eases off (highs, mids, a
    little level) as the incoming builds. After the bass swap it fades out
    over 2 to 8 more bars, so it never just disappears.
  - **Echo-out is kept for real clashes** (keys far apart, or a tempo jump of
    more than 10 %). Everything else gets a blend.
  - **Blend length depends on genre:** long blends for house and techno,
    short ones for hip-hop and pop. Your bars setting is the maximum.
  - **A peak limiter guards the master**, so two loud tracks overlapping
    never distort.
- **Auto-mix timing:**
  - **Auto** (the default) picks the blend length from the music, up to the
    bars you set. With Auto off, every blend lasts exactly that long (4, 8,
    16, 32, 64 or 128 bars, or any value from 1 to 128).
  - **Queue:** drag tracks to reorder them. To remove one, click its ✕,
    press Delete, or drag it out of the list. **Clear queue** empties it.
  - Auto-mix starts the blend at the outgoing track's outro marker and
    brings the next track in at its intro marker.
  - Without markers, it ends the blend on the last bar that still has
    beats and starts the next track on its first beat, so there is no dead
    air.
- **2 beat-synced FX units:** Delay, Reverb, Flanger, Gater, Crush, each
  with dry/wet, amount and beat timing. The units are inserts that you
  assign per channel.
- **📱 Phone remote (mix server):** click **📱 Phone remote** and scan the
  QR code with your phone (same Wi-Fi). The phone shows the album art, the
  two decks with spinning records and the mixer, and the full queue and the
  tracks played tonight. From the phone you can:
  - skip (mixed in over a few bars), **cut now**, go back, and pause/resume
    the whole mix (it fades);
  - search the library and pick a **seed song**: CTA builds a new set from
    it and mixes it in;
  - play a track next, add, reorder or remove tracks in the queue;
  - edit the **IN / OUT / OUT END markers**: pinch or tap ＋/− to zoom,
    swipe to scroll, then tap a bar, drag a flag or nudge it −4/+4 beats.
    You can also set the outro length (16–128 bars) and the blend length
    (Auto, 4–128 bars);
  - swipe a queued track left to remove it, or **Clear queue**;
  - tap **Listen** to hear the mix. **Live** (the default) is about 0.2 s
    behind the PC but stops when the phone's screen locks. **Radio** keeps
    playing with the screen locked and runs a few seconds behind.

  On iPhone, use Share → **Add to Home Screen**: it opens full screen like
  an app, with its own icon, and connects by itself. A 6-digit PIN (shown
  with the QR code) keeps others on the network out; wrong guesses are
  locked out. Away from home, use a VPN such as Tailscale. Windows will ask
  once to allow CTA through the firewall.
- **Server mode:** in Settings, tick *Start CTA in the background when I
  log in*. CTA then starts hidden in the system tray with the phone remote
  running, so the PC is a mix server or radio station you control from the
  phone. Closing the window keeps it serving; quit from the tray icon.
  Without a sound card the mix still runs for the phone stream and
  recordings.
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
  - The whole unit works straight away, from Pioneer's MIDI list: play, cue,
    jog (touch to scratch), tempo and TEMPO RANGE, KEY LOCK, loop IN / OUT /
    ACTIVE, AUTO LOOP (press: loop, turn: size), FILTER, the needle-search
    strip (stopped decks), the four performance buttons (HOT CUE mode: hot
    cues 1-4, SHIFT 5-8, hold DELETE to remove one; MOVE mode: loop ½ / ×2,
    BWD / FWD), and the whole mixer: GAIN, EQ, channel faders, crossfader,
    headphone CUE / MIX / VOL, SYNC, MASTER, FX assign, LOAD, browse and the
    FX dry/wet, parameter and on buttons.
  - **Import Traktor (.tsi)…** in Controller setup reads a Traktor settings or
    mapping export and sets up the DDJ-T1 from it (listing anything CTA has
    no equivalent for).
    **Controller setup…** then walks you through every control once ("move
    the deck A tempo fader…"). CTA shows what it heard and waits: press
    **Space** to keep it and go on (**Backspace** listens again, **S** or
    Space with nothing heard keeps what is set). Steps you don't confirm keep
    their old settings, so a fader that keeps sending can't run through the
    list. **Export…** saves the mapping to a file and **Import…** loads one
    (it is then used every time that controller connects). LEDs follow the
    deck state.
  - Jog wheel: touch the top to scratch, turn the outer ring to nudge, or
    search when paused. **Options → Scratch** turns scratching off (jog top,
    platter and waveform drag); the jog then only nudges or searches.
  - **Reset to built-in** in Controller setup puts back CTA's full DDJ-T1
    map; a saved mapping missing controls (jog touch, the mixer…) is
    completed from it automatically.
  - Any other MIDI controller can be mapped with the same wizard.
- **LOOP OUT** with no IN set loops the chosen loop length at once; after an
  IN it loops from the IN point to where you are.
- **Jog nudge on a synced deck stays put:** the sync lock keeps the phase
  where your nudge left it (press SYNC again to snap back on the beat).
- **CUE / PLAY start on the first sample** at the track's own tempo (no
  time-stretcher warm-up), so fast cue taps are exact.
- With a **controller attached**, loading a track keeps the gain / EQ /
  filter where your knobs are (without one they reset to neutral).
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

## If something goes wrong

CTA writes two files to its data folder (on Windows
`%LOCALAPPDATA%\CTA`, on macOS `~/Library/Application Support/CTA`):

- `cta.log`: what CTA was doing, including errors.
- `crash.log`: written if CTA itself crashes (for example in an audio
  driver). CTA says so on the next start.

If the sound card stops responding, CTA reopens it by itself within a few
seconds. The phone remote shows a red banner when the PC's mix has stopped
or the phone has lost the connection.

## Updates

CTA keeps itself up to date. At start-up (at most every 12 hours) it checks
GitHub Releases. When a new version is out, you can **Update now**, **Later**
or **Skip this version**. You can also click **Check for Updates** in the
toolbar at any time, or turn automatic checks off in Settings.

**Mac:** an app opened straight from Downloads runs from a hidden, read-only
copy, which can't be updated in place. CTA then installs the update to
**/Applications** (or **~/Applications**) and clears macOS's quarantine flag,
so from then on it updates itself where it is.

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
