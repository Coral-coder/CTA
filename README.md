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
   CTA starts in **DJ mode** (decks above the library); switch that off
   under Settings if you only want the library.
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

- **Double-click a track** for its **details page**: the numbers, how it
  feels (energy, tone, busyness, style family), how it would sit after
  what is playing (energy step, tempo change, key, style fit), a map of
  its sections with the energy per bar and the first drop, and the full
  waveform with its mix markers. Load it to a deck or add it to the
  setlist from there. (The external player is still in the right-click
  menu.)

## DJ Mode (Ctrl+D)

The decks sit above the library in a compact, Traktor-style layout. Drag
the bar between them to give the decks more or less room.

- **Global row (top):** FX 1, the auto-mix clock (master BPM, Auto-mix,
  skip, pause, blend length), the MAIN level and limiter, FX 2 and the
  recorder. Hide it under **Options**. The auto-mix queue is a tab next to
  the library.
- **Waveform layout (Options):** *In each deck* (classic), or **Stacked
  across the top (Serato style)** - every deck's zoomed waveform rides in
  parallel in one panel above the decks with the playheads in one line, so
  the beats of A, B (and C, D) line up visually; the decks keep their
  overview strip, platter and tempo fader. The *Three-band* colours look
  most like Serato.
- **Four decks in auto-mix:** with **Decks C/D** on, auto-mix uses all four
  round the table (A → B → C → D → A) - the next track never lands on the
  deck that just finished, so you can keep working a track while the mix
  carries on.
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
- **⏏ Unload decks:** with auto-mix stopped and its queue empty, the status
  bar offers to empty the decks that aren't playing and reset auto-mix (queue,
  plan, seed, what it played in that run) for the next set. The auto-mix ▾
  menu also has **Unload stopped decks** and **Unload all decks** (asks
  first: it stops the music). What auto-mix learned from you and your Mix
  Ahead recordings are kept.
- **Every control drags from where it is:** knobs (up = more, down = less),
  channel faders, the crossfader and the tempo faders. A click never jumps a
  control to the mouse; hold Shift for fine moves.
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
  - **Just mixing records:** with **Record whenever I mix the two decks** on,
    starting the incoming while the outgoing plays records the transition
    (saved for auto-mix by itself). Turn it off to simply mix and listen.
  - **Controls I didn't touch follow auto-mix** (on by default): only what you
    moved is replayed, auto-mix does the rest (its bass swap, the fades).
    Your crossfader moves really move the crossfader.
  - A mix recorded after auto-mix already planned that transition is still
    used, as long as the transition has not started yet.
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
- **The audio engine runs in its own process.** Decks, mixing, auto-mix,
  the sound card, extra speakers and the Mix Ahead preview live in a child
  process with nothing else to do; the window talks to it over a pipe
  (commands one way, 30 state snapshots a second the other) and the music
  comes back through shared memory for the phone stream, the recorder and
  the scopes. The screen painting, library scans, a stuck dialog or a slow
  plugin can no longer delay a single audio block: in the stress test the
  window's interpreter is frozen 150 ms out of every 200 and the audio
  process stays exactly on time. If the audio process ever dies, the window
  starts a new one and reloads the decks (watchdog). Controls still land
  within a millisecond. Audio… has a switch to run it in-process instead.
- **Audio first.** The audio engine runs in its own process at high
  priority; its callback thread is made the most important thread on the
  machine (Windows: time-critical + the MMCSS "Pro Audio" class every DAW
  uses; macOS: a real-time time-constraint policy sized to the block; Linux:
  SCHED_FIFO when allowed). The GUI comes second: nothing it does - paints,
  scans, downloads - can delay a block.
- **Audio… → Output safety:** how much audio the sound card holds in
  reserve. **Auto** (default) lives on the edge of safe: it starts at the
  card's minimum, steps the reserve up one notch the moment a block is
  late (12 → 25 → 50 → 100 ms) and steps back down after ten quiet minutes
  with nothing playing - the status bar says when it moves. **Normal**
  (25 ms), **Safe** (50 ms) and **Tight** (the minimum, no safety net) fix
  it instead. CTA also keeps full speed in the background: no App Nap on
  the Mac, no power throttling on Windows, and the time-stretcher is never
  rebuilt inside an audio block.
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
  - **⚑** on each deck (or right-click the waveform → Edit markers) opens
    a zoomable full waveform with bar numbers and a strip of the song's
    sections (intro, groove, build, drop, breakdown, outro). **Add a
    marker:** press **+ IN**, **+ OUT** or **+ OUT END**, then click where
    it goes (the start of that bar). **Remove:** the **− IN / − OUT /
    − OUT END** buttons, or **Remove** on the marker's row. Drag a flag to
    move it, nudge it **−4 / +4 beats**, or **Set at playhead**. The
    **outro length** buttons (8, 16, 32, 64 or 128 bars) set OUT END for
    you; past the end of the track, auto-mix loops the groove to last that
    long. **Clicking the waveform never moves the deck** - "Play from here"
    is in the right-click menu. The intro (start → IN) is shaded green and
    the outro blend (OUT → END) red, in the editor and on the deck's
    overview strip.
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
- **Learned mixes & templates** (auto-mix menu): every learned transition
  in a table you can **perfect** (edit bars, out offset, in lead, style),
  delete, or **make the template** for its style - from then on every
  transition of that style (House, Trance, Drum & Bass, …) is mixed with
  exactly those numbers, no averaging. A **Mix Ahead recording** can be the
  template too: pick it, choose the style, and its fader and EQ moves are
  replayed for every transition of that style, timed by the planner for
  the tracks at hand. **Export…** writes learned mixes, templates and
  recordings to one JSON file; **Import…** merges such a file (from a
  backup or another machine).
- **Style families (EDM sub-genres):** the AI's 400 styles are grouped
  into House, Disco, Techno, Trance, Bass (dubstep, garage, trap), Drum &
  Bass, Breaks, Hard Dance, Dance / Pop and Downtempo, with a map of
  which families **blend** with each other (house ↔ techno ↔ hard dance,
  bass ↔ drum & bass ↔ breaks, …). Suggestions, the auto-mix queue and
  the Setlist Builder stay in the seed's family or one that blends with
  it (**Style: Family + what blends with it**, the default; or *Same
  style family only* / *Any style*). The library gets a **Style** filter
  (AI sub-genre, narrowed by the Genre filter), the track line shows the
  family and its neighbours, and **What blends?** opens the whole map.
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
  - **Each style family is mixed its own way:** trance and drum & bass
    never shorter than 16 bars with the bass swap on the incoming drop at
    the very end of the blend; dubstep / bass and hard dance 8 bars or
    more, also swapping on the drop; house, techno, disco and breaks as a
    classic EQ blend with the swap on a phrase in the middle. When the
    incoming drop lands inside the blend, the basses swap right there. A
    short outro never cuts a family's minimum: the last groove is looped.
  - **The bass comes in gradually:** the incoming low end creeps in
    through the second half of the build-up (to 60 % by the downbeat), the
    outgoing gives it the same room, and the downbeat finishes the swap.
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
- **Taking over from auto-mix:** touching the incoming deck's fader, EQ or
  filter *before* a planned blend, or anything on either deck *during* one,
  makes that transition yours: auto-mix never starts or changes it over your
  settings. **Options → Auto-mix carries on after I take over** (on by
  default) brings auto-mix back once your blend is done; off, it stays off
  until you press ▶ AUTO-MIX again. A track you mixed out of by hand is never
  mixed back into.
- **CUE** previews straight from the cue point: sync no longer pulls a
  previewing deck, and the playhead never draws a step back when you press
  CUE or PLAY.
- **Drag on the overview strip** (the whole-track waveform) to scrub.
- **Forget what was learned** asks whether to drop the Mix Ahead recordings
  too, and re-plans the next transition; learned blends are never shorter
  than 4 bars, and a blip (a deck started and stopped at once) is never
  learned or recorded.
- **What you see is what you hear:** the playhead, cue point, hot cues and
  loop points line up with the sound. The waveform is drawn as late as the
  output plays (the driver's figure, or what you measured); CUE, hot cues,
  loop IN / OUT and "beat 1 here" pressed while playing land on what you
  *heard* at that moment, not on what was about to be rendered. A delay you
  measured (or the Bluetooth guess) belongs to that device only - it is
  never applied to the wired card you play on later (Audio… shows "measured"
  or "(driver)"). With the engine in its own process the GUI runs the
  positions on smoothly between the engine's 30 updates a second.
- **SYNC on both decks:** the master deck leads and its tempo fader moves
  the whole mix; the other follows. A deck holding CUE is never the deck
  others follow, so cueing one deck never bends the other.
- **Tempo sync or beat sync:** **Options → SYNC also locks the beat phase**.
  Off: SYNC matches the tempo only and you line up the beats yourself (they
  stay where you put them; nothing jumps when you press PLAY or SYNC). On:
  tempo and beat. Auto-mix keeps its own blends on the beat either way.
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

- **If the audio process ever crashes**, the watchdog starts a new one and
  puts everything back where it was - every deck's track, position, play
  state, cue, hot cues, loop and levels, the master deck and the auto-mix
  queue - so it costs a moment of silence, not the mix. The status bar says
  "Audio process restarted" and `audio.log` in the data folder has the
  reason.

## Library bridge: your home collection, anywhere

Your main collection lives on the computer at home; the laptop you take
to a gig has CTA too. **Settings → Library bridge** connects them:

1. **At home:** tick **Share this library**, pick a port (8008) and put in
   the address the laptop will reach you at (a hostname or your public IP
   with the port forwarded on your router - or a tunnel such as
   [Tailscale](https://tailscale.com), which gives you a stable address
   and needs no router changes). **Copy** the pairing code. Sharing keeps
   running with the mix server in the tray.
2. **On the laptop:** paste the code under **Home libraries on this
   machine** and press **Add**. CTA contacts home, checks the key, and
   fetches the whole catalogue: every track with its tags, BPM, key, AI
   style and sound vectors, beat grid, cues and mix markers - nothing is
   re-analysed. The tracks appear in your library marked **⬇** (blue)
   until downloaded; search, suggestions, the Setlist Builder and the
   auto-mix treat them like your own.
3. **Playing:** load one to a deck (or let the auto-mix queue it) and it
   is downloaded first - the deck title shows the progress; anything in
   the auto-mix queue is fetched ahead of time. Right-click → **Download
   now** (one track, a selection, or the whole setlist) to fetch before
   the gig. Downloads land in the data folder under `remote/` and stay;
   a download that was cut off resumes.
4. **Sync now** refreshes the catalogue (new tracks at home show up; ones
   deleted at home disappear here); it also runs at start-up. **Remove**
   forgets a home library (optionally deleting its downloads).

Security: the key is 32 random bytes and the connection is encrypted with
a certificate the laptop pins from the pairing code (no certificate
authority involved); wrong keys are rate-limited. **New key** at home
revokes every paired laptop.

## Updates

CTA keeps itself up to date. At start-up (at most every 12 hours) it checks
GitHub Releases. When a new version is out, you can **Update now**, **Later**
or **Skip this version**. You can also click **Check for Updates** in the
toolbar at any time, or turn automatic checks off in Settings.

**Signed and notarized Mac builds.** When the repository has the signing
secrets (`DEVELOPER_ID_P12`, `DEVELOPER_ID_P12_PASSWORD`, and the App Store
Connect key `ASC_KEY_ID` / `ASC_ISSUER_ID` / `ASC_KEY_P8` — all handed out by
the **Dev_deploy_setup** hub: *Create Developer ID certificate* and *Set up
app repo*, `repos` = `CTA`), every release is signed with the hardened runtime,
notarized by Apple and stapled: it opens without the right-click dance and
updates itself in place. The updater then only accepts a new version signed
by the same team. Without the secrets the build is unsigned, as before.

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
