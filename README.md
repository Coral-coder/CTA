<h1 align="center">CTA</h1>

<p align="center">
  <b>The DJ library that knows your music - and mixes it.</b><br>
  BPM · key · energy · AI style for every track · next-track suggestions in a millisecond ·
  setlists in a second · a four-deck DJ mode with an auto-DJ that learns how <i>you</i> mix.
</p>

<p align="center">
  <a href="https://github.com/coral-coder/CTA/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/coral-coder/CTA?label=download&color=2ea043"></a>
  <a href="https://github.com/coral-coder/CTA/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/coral-coder/CTA/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/platforms-Windows%2010%2F11%20·%20macOS%20(Apple%20Silicon)-1f6feb">
  <img alt="Licence" src="https://img.shields.io/badge/licence-GPL%20v2%2B-blue">
</p>

---

## Why CTA

- **It hears what tags can't tell you.** Every track is analysed once: BPM to 0.005, key with its Camelot code, energy, and the AI's take on genre and style (400 Discogs styles, plus a "sounds like" fingerprint). Tech House is not Deep House.
- **It answers instantly.** The whole library lives in memory as vectors: the best next track in about 1 ms, a 40-track setlist from 50 000 tracks in well under a second.
- **It DJs like a DJ.** Auto-mix reads each track's structure (intro, build, drop, breakdown, outro), mixes out after the last drop, lands the next drop on the "one", swaps the bass on the downbeat, loops when the outro is short, echoes out on a key clash - and it learns from every transition you do by hand.
- **Audio comes first.** The engine runs in its own process at real-time priority with a buffer that lives on the edge of safe. Nothing the window does - paints, scans, downloads - can delay a block.
- **Built to stay up.** Decoder crashes are isolated in worker processes; SQLite in WAL mode; scans stop and resume; a crashed audio process is restarted with the whole mix put back where it was.

## Contents

- [Download](#download)
- [Five minutes to a set](#five-minutes-to-a-set)
- [The library](#the-library)
- [DJ mode](#dj-mode)
- [Auto-mix](#auto-mix)
- [Audio engine and outputs](#audio-engine-and-outputs)
- [Controllers](#controllers)
- [Phone remote and server mode](#phone-remote-and-server-mode)
- [Library bridge: your home collection, anywhere](#library-bridge-your-home-collection-anywhere)
- [Updates](#updates)
- [Under the hood](#under-the-hood)
- [If something goes wrong](#if-something-goes-wrong)
- [Command line](#command-line)
- [Development](#development)
- [Licences](#licences)

## Download

Grab the latest build from the [Releases](https://github.com/coral-coder/CTA/releases) page. The AI model is bundled; nothing else needs installing.

| Platform | File | Run |
|---|---|---|
| Windows 10 / 11 x64 | `CTA-x.y.z-windows-x64.zip` | unzip, run `CTA\CTA.exe` |
| macOS (Apple Silicon) | `CTA-x.y.z-macos-arm64.zip` | unzip, right-click `CTA.app` → **Open** (first time only; signed builds open directly) |

CTA keeps itself up to date from then on - see [Updates](#updates).

## Five minutes to a set

1. **Add Folder…** and point CTA at your music. The scan starts by itself, in parallel across every core; the AI runs on the GPU (DirectML on Windows - any DX12 card; CoreML on the Mac). Re-scans only touch new or changed files.
2. CTA opens in **DJ mode** (decks above the library). Click any track: the **Next Track** tab ranks every compatible track by tempo, key, sound, genre and energy.
3. Drag a track to a deck (or right-click → **Load to Deck A**), press **PLAY**.
4. Queue a few tracks (right-click → **Queue in auto-mix**, or just tick *Auto-load best next track*) and press **▶ AUTO-MIX**. It beat-matches, phrase-aligns and blends them.
5. Take over whenever you like - touch a fader, EQ, the crossfader or a deck. Auto-mix steps back, learns what you did, and carries on when you're done.
6. Pair your phone (**📱 Phone remote**) to skip, search, seed a new set or fix markers from the floor.

## The library

**Scan & analysis.** BPM (onset autocorrelation + harmonic DFT, refined to 0.005 BPM), key (HPCP chroma, Shaath/KeyFinder profiles), loudness, energy, AI style (Discogs-EffNet, 400 styles) and a 1280-d sound embedding. Half/double-time mistakes are fixed from the style (Drum & Bass sits at 174, not 87). Right-click a track to double/halve its BPM or set the key by hand; corrections survive re-scans.

**Next Track.** From the focused track, every other track is scored on

- **Tempo** - how far the pitch fader would have to move, within your ± range; half and double time count (87 ↔ 174).
- **Key** - Camelot wheel rules: same key, ±1, relative major/minor, diagonal and energy-boost moves.
- **Sound** - similarity of the AI embeddings ("sounds like").
- **Genre** - the AI style profiles, and the style-family map (below).
- **Energy** - hold, lift or drop the level.

Weights and limits are yours to set. Double-click a suggestion to *follow* it and walk a set live.

**Style families.** The 400 AI styles are grouped into House, Disco, Techno, Trance, Bass (dubstep, garage, trap), Drum & Bass, Breaks, Hard Dance, Dance / Pop and Downtempo, with a map of which families blend (house ↔ techno ↔ hard dance, bass ↔ drum & bass ↔ breaks, …). Suggestions, the auto-mix queue and the Setlist Builder stay in the seed's family or one that blends with it (**Style: Family + what blends with it** - or *Same family only* / *Any style*). The **Style** filter narrows the list to one AI sub-genre; **What blends?** shows the whole map; the track line names the family and its neighbours. A lock never starves a set: when a family runs dry it widens.

**Sets stay in the seed's genre.** Every suggestion is also scored against the track the set started from, so a set that began with electro house can't drift into trance one "close enough" step at a time.

**Setlist Builder.** Length, an optional tempo ramp (122 → 128), an energy curve (steady / build / peak / wave / cooldown) and a variety amount; a beam search finds the sequence with the best transitions overall.

**Playlists.** Drag tracks in or right-click → *Add to playlist*. Between every two songs CTA shows a dimmed **bridge** suggestion - the track that best links them; double-click to keep it, right-click for another. *Queue in auto-mix* plays a playlist as a mix.

**Details page.** Double-click any track: the numbers, how it feels (energy, tone, busyness, family), how it would sit after what is playing (energy step, tempo change, key, style fit), a map of its sections with the energy per bar and the first drop, and the full waveform with its mix markers. Load it to a deck or add it to the setlist from there. (Double-click never opens an external player; that stays in the right-click menu.)

**Never the same song twice.** Auto-mix and auto-load skip anything played this session, queued, loading or on a deck - including other files of the same song. The **Played** column shows `● now` (this session) or `✓ 3×` (before); played tracks are dimmed.

**Export & drag.** Setlists and history export to **M3U8** (Serato, Traktor, VirtualDJ, Engine DJ, rekordbox), **rekordbox XML**, **Traktor NML** (with BPM and key) and **CSV**. Tracks dragged out of any list carry their audio files, so they drop straight onto a Traktor deck, a rekordbox playlist, Finder or Explorer.

## DJ mode

`Ctrl+D` toggles it. The decks sit above the library in a Traktor-style layout; drag the divider to give them more room. CTA fits a 1280-pixel laptop screen, and **Settings → Interface size** scales everything 70–150 %.

### Layout

- **Global row:** FX 1 · the auto-mix clock (master BPM, ▶ AUTO-MIX, ⏱ MIX AHEAD, skip, pause, blend length) · MAIN level with limiter and oscilloscope · FX 2 · recorder. Hide it under **Options**.
- **Waveform layout (Options):** *In each deck* (classic) or **Stacked across the top (Serato style)** - every deck's zoomed waveform rides in one panel with the playheads in a line, so the beats of A, B, C and D line up visually. The *Three-band* colours look most like Serato.
- **Scalable decks:** each deck shows what fits its height - **Full** (waveform, hot cues, loops, beat jump, key and grid tools, tempo fader, jog platter), **Essential** or **Micro**; *View: Auto* switches by itself. The mixer sheds knobs as it shrinks.
- **Decks C/D** (Options) - and auto-mix uses all four.

### Decks

- **Transport:** PLAY, **CUE** (reacts on press; hold to preview, drag off or press PLAY while holding and it keeps playing), CUP, FLX (flux: after a loop or jump the track carries on where it would have been).
- **Loops:** auto loops 1–32 beats, IN / OUT (OUT with no IN loops the chosen length at once), ½ / ×2, beat jump.
- **8 hot cues** per track, saved. **Key shift** ±12 semitones and **KEY SYNC** (the smallest transposition that makes the deck harmonic with the master). Beat-grid nudge and *set beat 1 here*, saved per track.
- **Waveforms** coloured by frequency - **Rainbow** (default), **Spectrum**, **Three-band**, **Fire**, **Ice**, **Classic**, with a saturation slider - plus a beat grid, a scrolling zoom (**− / +**, **⛓ sync zoom**) and the whole-track overview (drag it to scrub).
- **Scratching:** drag the waveform or grab the spinning platter (the album art turns on the record); let go and it carries on. **Options → Scratch** turns it off. **XY scope on the platters** draws each deck's stereo image.
- **Every control drags from where it is** - knobs, faders, crossfader, tempo fader. A click never jumps a control to the mouse; Shift = fine; double-click resets (crossfader to centre).
- **What you see is what you hear.** The playhead, cue point, hot cues and loop points line up with the sound: positions are drawn as late as the output plays, and CUE / hot cues / loop IN-OUT / beat 1 pressed while playing land on what you *heard* at that moment.

### Mix markers

Three per track, snapped to the bar and saved: **IN** (where it's mixed in), **OUT** (where mixing out starts) and **OUT END** (where it's gone - fixes the blend length). When a track loads, its cue point sits on the **first real beat**.

**⚑** on each deck opens the marker editor: a zoomable full waveform with bar numbers and a strip of the song's sections. Press **+ IN / + OUT / + OUT END** and click where it goes; **−** buttons remove; drag a flag, nudge **−4 / +4 beats**, or **Set at playhead**; **outro length** buttons (8–128 bars) set OUT END for you - past the end of the track, auto-mix loops the groove to last that long. The intro is shaded green, the outro blend red. **Clicking the waveform never moves the deck** (*Play from here* is in the right-click menu). Right-clicking a deck waveform sets markers on that bar too.

### Keyboard (Traktor Pro's default keys)

Deck A on the top letter row, deck B on the home row; never while you type in a text box. **Options → Keyboard shortcuts…** lists them.

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
| Load selected track | Shift+← (or Ctrl+←) | Shift+→ (or Ctrl+→) |
| Load to deck C / D | Ctrl+Shift+← | Ctrl+Shift+→ |

### Mixer and FX

Per channel: GAIN, HI / MID / LOW (kill when fully left, Linkwitz-Riley crossovers), a **filter** knob (left = low-pass, right = high-pass), FX 1 / 2 assign, headphone cue, a meter and the fader. Below: crossfader, master level and meter with a peak limiter, headphone mix and volume, and **REC** (24-bit WAV). Gain, EQ, filter and key reset when a new track loads - unless a controller is attached, then the knobs keep their place. Two beat-synced **FX units** (Delay, Reverb, Flanger, Gater, Crush) with dry/wet, amount and beat timing.

### Sync

- **SYNC** matches tempo (half/double time included) and phase-locks to the master; the phase meter shows how close. **Options → SYNC also locks the beat phase** off: tempo only, you line up the beats yourself and they stay where you put them.
- With SYNC on both decks the **lead is sticky**: the deck already playing keeps leading and its tempo fader moves the whole mix; a deck you CUE and PLAY never grabs the lead, so nothing pitch-bends to re-align. **MASTER** moves it.
- A jog nudge on a synced deck stays where you put it (SYNC again snaps back on the beat). CUE and PLAY start on the first sample at the track's own tempo - fast cue taps are exact.
- **Key lock** (Rubber Band R3, as in Mixxx, ~3 % of a core per deck) keeps the key while the tempo changes; off, you get vinyl-style pitch.

## Auto-mix

Press **▶ AUTO-MIX** with tracks queued (or *Auto-load best next track* on) and CTA plays a continuous mix. **Transition now** blends into the other deck from the next bar whenever you ask.

### How it DJs

- **It reads the structure.** Every phrase is labelled intro, groove, build, drop, breakdown or outro from the kick, bass, mid and high energy per bar; beat 1 is found from where those changes land, so blends are bar-for-bar on the right beat. Hand-edited grids are never touched.
- **It mixes out in the outro**, after the last drop has fully played, never cutting a drop or breakdown mid-phrase, and **mixes in from the next intro** timed so that track's **drop lands exactly when the blend ends**.
- **The bass swap happens on that downbeat.** The incoming bass creeps in through the second half of the build-up (to 60 % by the "one"), the outgoing gives it the same room, the downbeat finishes the swap. Highs come first, mids under; the outgoing eases off and fades over 2–8 more bars after the swap - a real crossfade, never a drop-out.
- **Each style family is mixed its own way:** trance and drum & bass never shorter than 16 bars with the swap on the drop at the very end; dubstep / bass and hard dance 8 bars or more, also on the drop; house, techno, disco and breaks a classic EQ blend with the swap on a phrase in the middle; hip-hop and pop short. A drop landing inside the blend is the swap point.
- **It loops to keep the beat** when the outro is short or missing, layers a long beatless intro over the outgoing groove (and a long outro fill over the next beat, intro looped), and **never hangs on dead air** - every bar of the blend is checked for music and a beat.
- **It echoes out on a real clash** (keys far apart, tempo gap over 10 %) and drops the next track clean on the one. Everything else gets a blend.
- **Blend length:** *Auto* (default) from the music, up to your bars; or exactly 1–128 bars. Afterwards the tempo glides to the new track's own BPM.
- **Four decks:** with Decks C/D on, the next track goes round the table (A → B → C → D → A) - never onto the deck that just finished, so you can keep working it.

### It learns from you

Every transition you mix by hand is a lesson: where in each track's structure you blended, for how long, how far before the drop you brought the next one in. **Taking over** an auto-mix transition (any fader, EQ, crossfader, play, cue, tempo, loop or jog move, by mouse, keyboard or controller - or setting up the incoming deck *before* a planned blend) makes it yours at once; auto-mix carries on by itself a few seconds after your blend is done (**Options → carries on after I take over**, or stays off). After three examples for a style, that style is mixed your way.

**Learned mixes & templates** (auto-mix ▾ menu): every learned transition in a table you can perfect (bars, out offset, in lead, style), delete, or **make the template** for its style - from then on every transition of that style uses exactly those numbers. A **Mix Ahead recording** can be the template too: its fader and EQ moves are replayed for every transition of that style, timed by the planner for the tracks at hand. **Export / Import** carry learned mixes, templates and recordings as one JSON file. **Forget** drops it all.

### ⏱ Mix Ahead (Time-Shift)

Mix the *next* transition now, on headphones, while the room hears the live decks. A separate preview player holds the outgoing and incoming tracks as zoomable full waveforms (place markers, scrub, listen) with its own mixer - or tick **My controller mixes the preview** and your hardware drives it. **● Rehearse** plays the outgoing from 8 bars before its OUT, drops the incoming in phase on the OUT marker (or **Drop incoming now**), and records every move on the beat until the outgoing is gone. Simply mixing the two preview decks records too (toggle). Controls you didn't touch follow auto-mix's own blend; your crossfader moves really move the crossfader. When the live set reaches those two tracks, auto-mix starts the incoming on the same beat and replays your moves.

### Queue and housekeeping

Drag to reorder; ✕, Delete or drag out to remove; **Clear queue**. With auto-mix stopped and the queue empty, the status bar offers to **unload the decks** that aren't playing and reset auto-mix (queue, plan, seed, this run's history) for the next set - what it learned is kept.

## Audio engine and outputs

**Its own process.** Decks, mixing, auto-mix, the sound card, extra speakers and the Mix Ahead preview run in a child process with nothing else to do. The window talks to it over a pipe (commands one way, 30 state snapshots a second back) and the music comes back through shared memory for the phone stream, the recorder and the scopes. In the stress test the window is frozen 150 ms out of every 200 and the audio process stays exactly on time. Controls still land within a millisecond. If the process ever dies, the watchdog starts a new one and puts **everything back** - tracks, positions, play state, cues, hot cues, loops, levels, master deck, queue - a moment of silence, not the mix.

**Audio first.** The audio process runs at high priority; its callback thread makes itself the most important thread on the machine on its first block (Windows: time-critical + the MMCSS "Pro Audio" class; macOS: a real-time time-constraint policy sized to the block; Linux: SCHED_FIFO when allowed). No App Nap on the Mac, no power throttling on Windows; the time-stretcher is never rebuilt inside a block.

**Output safety (Audio…).** **Auto** (default) lives on the edge of safe: it starts at the card's minimum buffer, steps up one notch the moment a block is late (12 → 25 → 50 → 100 ms) and steps back down after ten quiet minutes with nothing playing; the status bar says when it moves. **Normal** (25 ms), **Safe** (50 ms) and **Tight** (the minimum, no net) fix it instead.

**Outputs.** Devices are scanned live without stopping the music; a remembered speaker that reappears rejoins by itself, a newly paired Bluetooth device offers **Add … as output**. Master and headphones can go to any device (a DDJ-T1's outputs, the laptop jack, Bluetooth). **Extra master outputs** play the mix on any number of further devices, kept in sync.

**Clock.** Pick the output everything lines up with (or **Auto**: the slowest, usually Bluetooth), choose which outputs **sync to the clock** (untick headphones to hear them straight), and fine-tune by ear with **Clock adjust** while the music plays. Each output's delay is the driver's figure, ~200 ms for Bluetooth, or **measured** by tapping along to clicks; a measured delay belongs to that device only and is never applied to the wired card you play on later.

## Controllers

**Pioneer DDJ-T1** works out of the box: both MIDI ports and its 4-channel card (master 1-2, headphones 3-4). Play, cue, jog (touch to scratch, ring to nudge, search when paused), tempo and range, key lock, loops, auto loop, filter, needle search, the four performance buttons (hot cues / move modes), the whole mixer, SYNC, MASTER, FX assign, LOAD, browse and the FX section; LEDs follow the decks.

**Any MIDI controller:** **Controller setup…** walks you through every control once - it shows what it heard and waits for **Space** before moving on, so a fader that keeps sending can't run through the list. **Import Traktor (.tsi)…** sets a controller up from a Traktor export; **Export… / Import…** save and load mappings (used every time that controller connects); **Reset to built-in** restores the DDJ-T1 map.

## Phone remote and server mode

**📱 Phone remote:** scan the QR code (same Wi-Fi; a 6-digit PIN keeps others out). The phone shows album art, the decks with spinning records, the mixer, the queue and tonight's history. Skip (mixed in), **cut now**, go back, pause the whole mix (it fades), search and pick a **seed song** for a new set, play a track next, edit the queue, fix **IN / OUT / OUT END markers** with pinch-zoom and drag, set outro and blend lengths, and **Listen** - *Live* (~0.2 s behind) or *Radio* (keeps playing with the screen locked). On iPhone, *Add to Home Screen* makes it an app. Away from home, use a VPN such as Tailscale.

**Server mode:** *Start CTA in the background when I log in* runs it hidden in the tray with the phone remote up - a mix server or radio station you drive from the phone. Without a sound card the mix still runs for the stream and recordings. **History** logs every track that plays 30 s or more and exports as M3U8 / NML / XML / CSV.

## Library bridge: your home collection, anywhere

Your collection lives at home; the laptop you take out has CTA too. **Settings → Library bridge** joins them.

1. **At home:** tick **Share this library**, set the port (8008) and the address the laptop will reach you at (a hostname or public IP with the port forwarded - or a tunnel such as [Tailscale](https://tailscale.com), which gives a stable address with no router changes). **Copy** the pairing code. Sharing keeps running with the mix server in the tray.
2. **On the laptop:** paste the code under **Home libraries on this machine** → **Add**. CTA checks the key and fetches the whole catalogue - tags, BPM, key, AI style and sound vectors, beat grids, cues, markers - nothing is re-analysed. The tracks appear in your library marked **⬇** until downloaded; search, suggestions, setlists and auto-mix treat them as your own.
3. **Playing:** load one to a deck (or let auto-mix queue it) and it downloads first - progress on the deck title; the queue is fetched ahead of time. Right-click → **Download now** (a track, a selection, the whole setlist) before the gig. Downloads resume and stay in the data folder under `remote/`.
4. **Sync now** refreshes the catalogue (also at start-up); **Remove** forgets a home library, optionally with its downloads.

Security: a 32-byte key, TLS with a certificate the laptop pins from the pairing code (no certificate authority), rate-limited wrong keys; **New key** at home revokes every laptop.

## Updates

CTA checks GitHub Releases at start-up (at most every 12 hours) and on **Check for Updates**: **Update now**, **Later** or **Skip this version**. The zip is downloaded, verified against the release's SHA-256 sums, unpacked next to the install and test-run; only then does CTA close, a helper swaps the folders (keeping the old version until the swap has worked, rolling back if it fails) and restarts. Your library, settings and analysis data are never touched. On the Mac an app run straight from Downloads is installed to **/Applications** first so it can update itself in place.

**Signed Mac builds.** With the signing secrets in the repository (`DEVELOPER_ID_P12`, `DEVELOPER_ID_P12_PASSWORD`, and the App Store Connect key `ASC_KEY_ID` / `ASC_ISSUER_ID` / `ASC_KEY_P8`, handed out by the **Dev_deploy_setup** hub: *Create Developer ID certificate* and *Set up app repo*) every release is signed with the hardened runtime, notarized and stapled, and the updater only accepts a new version signed by the same team.

## Under the hood

| Task | Method | Why |
|---|---|---|
| BPM | DSP: onset autocorrelation + harmonic DFT, refined to 0.005 BPM | Accurate and fast for a beat grid |
| Key | DSP: HPCP chroma with tuning correction, Shaath/KeyFinder profiles | The standard DJ-software approach |
| Genre / style | **AI**: Discogs-EffNet, 400 styles | Tags are missing or wrong; the model knows Tech House from Deep House |
| "Sounds like" | **AI**: 1280-d embeddings | The next track matches the vibe, not only BPM and key |
| Half/double time | **AI style** + rhythm analysis | Drum & Bass at 174, not 87 |
| Beat 1 / phrases | Section changes in the music | Blends land bar-for-bar on the right beat |

The model is ~18 MB and uses well under 1 GB of VRAM. Decoding and DSP run on the CPU: 1–3 tracks per second per core; a 16-core desktop scans 10 000 tracks in 5–10 minutes, re-scans take seconds. The model input matches Essentia's reference to within 1e-3 (there is a unit test).

## If something goes wrong

The data folder is `%LOCALAPPDATA%\CTA` (Windows) or `~/Library/Application Support/CTA` (macOS) - Settings shows the path.

- `cta.log` - what CTA was doing, with errors.
- `audio.log` - the audio process (rewritten each launch); the reason for any "Audio process restarted".
- `crash.log` - written if CTA itself crashes (for example in an audio driver).

A sound card that stops responding is reopened within seconds; the phone shows a red banner when the mix or the connection has stopped. Dropouts on a weak machine: **Audio… → Output safety → Safe**.

## Command line

Windows builds include `cta-cli.exe`; on the Mac run `CTA.app/Contents/MacOS/CTA <command>`.

```
cta scan "D:\Music"                     # add a folder and analyse it
cta list --search "tech house"
cta next "artist - title"               # best next tracks
cta setlist "artist - title" -n 30 --end-bpm 128 --curve build -o set.m3u8
cta update --check                      # or: cta update
cta selftest                            # decoding, DSP, AI, DB, engine, stream: all checked
```

## Development

```
pip install -e ".[dev]"
pytest -q                 # the full suite (engine, auto-mix, GUI, audio process, bridge)
python -m cta             # start the GUI
pyinstaller packaging/cta.spec
```

CI (`.github/workflows/ci.yml`) lints and tests on Windows, macOS and Linux, builds both executables, runs the self-test and a GUI smoke test against the **frozen** build, and - on a release dispatch - signs/notarizes the Mac app when the secrets exist and publishes the release with both zips and checksums.

Versioning: patch releases (`x.y.Z`) ship fixes; minor (`x.Y.0`) new features; major (`X.0.0`) architecture changes.

## Licences

The bundled Discogs-EffNet model is **CC BY-NC-SA 4.0** (non-commercial) by MTG/UPF. Key lock uses **Rubber Band (GPL v2+)**, so the distributed builds are GPL-licensed. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
