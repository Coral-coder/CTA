"""Setlist export: M3U8 (Serato, Traktor, VirtualDJ, Engine, rekordbox),
rekordbox XML (with BPM/key) and CSV."""

from __future__ import annotations

import csv
import os
import pathlib
import urllib.parse
from xml.sax.saxutils import quoteattr

from .keys import camelot_str, key_name
from .library import Library


def write_m3u8(lib: Library, seq: list[int], path: str, name: str = "Tempest setlist"):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("#EXTM3U\n")
        f.write(f"#PLAYLIST:{name}\n")
        for i in seq:
            t = lib.track(i)
            label = f"{t.artist} - {t.title}" if t.artist else t.title
            f.write(f"#EXTINF:{int(round(t.duration))},{label}\n")
            f.write(f"{os.path.abspath(t.path)}\n")


def write_csv(lib: Library, seq: list[int], path: str):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "Artist", "Title", "BPM", "Key", "Camelot", "Energy", "Genre", "Duration", "Path"])
        for n, i in enumerate(seq, 1):
            t = lib.track(i)
            w.writerow([n, t.artist, t.title, f"{t.bpm:.2f}", key_name(t.key), camelot_str(t.key),
                        f"{t.energy:.1f}", t.genre or t.tag_genre,
                        f"{int(t.duration // 60)}:{int(t.duration % 60):02d}", t.path])


def _location(path: str) -> str:
    uri = pathlib.Path(os.path.abspath(path)).as_uri()          # file:///C:/... or file:///Users/...
    # rekordbox expects file://localhost/<path>
    rest = uri[len("file://"):]
    return "file://localhost" + urllib.parse.quote(urllib.parse.unquote(rest), safe="/:")


def write_rekordbox_xml(lib: Library, playlists: dict[str, list[int]], path: str):
    """rekordbox XML: import via Preferences > Advanced > rekordbox xml."""
    used = sorted({i for seq in playlists.values() for i in seq})
    tid = {i: n + 1 for n, i in enumerate(used)}
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<DJ_PLAYLISTS Version="1.0.0">',
             '  <PRODUCT Name="Tempest" Version="1.0" Company="Tempest"/>',
             f'  <COLLECTION Entries="{len(used)}">']
    for i in used:
        t = lib.track(i)
        lines.append(
            f'    <TRACK TrackID="{tid[i]}" Name={quoteattr(t.title)} Artist={quoteattr(t.artist)} '
            f'Album={quoteattr(t.album)} Genre={quoteattr(t.genre or t.tag_genre)} '
            f'TotalTime="{int(round(t.duration))}" AverageBpm="{t.bpm:.2f}" '
            f'Tonality={quoteattr(key_name(t.key))} Location={quoteattr(_location(t.path))}/>')
    lines.append("  </COLLECTION>")
    lines.append("  <PLAYLISTS>")
    lines.append(f'    <NODE Type="0" Name="ROOT" Count="{len(playlists)}">')
    for name, seq in playlists.items():
        lines.append(f'      <NODE Name={quoteattr(name)} Type="1" KeyType="0" Entries="{len(seq)}">')
        for i in seq:
            lines.append(f'        <TRACK Key="{tid[i]}"/>')
        lines.append("      </NODE>")
    lines.append("    </NODE>")
    lines.append("  </PLAYLISTS>")
    lines.append("</DJ_PLAYLISTS>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _nml_key(key: int) -> str:
    """Traktor's MUSICAL_KEY value: 0-11 = C..B major, 12-23 = Cm..Bm."""
    return str(key) if key is not None and key >= 0 else ""


def write_traktor_nml(lib: Library, seq: list[int], path: str, name: str = "Tempest setlist"):
    """Traktor Pro playlist (.nml): import via Traktor's browser
    (right-click Playlists > Import Playlist)."""
    entries, keys = [], []
    for i in seq:
        t = lib.track(i)
        p = os.path.abspath(t.path)
        drive, rest = os.path.splitdrive(p)
        folder, fname = os.path.split(rest)
        dir_nml = "/:" + "/:".join(x for x in folder.replace("\\", "/").split("/") if x) + "/:"
        volume = drive or ""
        key = f"{volume}{dir_nml}{fname}"
        keys.append(key)
        entries.append(
            f'<ENTRY TITLE={quoteattr(t.title)} ARTIST={quoteattr(t.artist)}>'
            f'<LOCATION DIR={quoteattr(dir_nml)} FILE={quoteattr(fname)} VOLUME={quoteattr(volume)} VOLUMEID=""/>'
            f'<ALBUM TITLE={quoteattr(t.album)}/>'
            f'<INFO GENRE={quoteattr(t.genre or t.tag_genre)} PLAYTIME="{int(round(t.duration))}" '
            f'KEY={quoteattr(camelot_str(t.key))}/>'
            f'<TEMPO BPM="{t.bpm:.6f}" BPM_QUALITY="100.000000"/>'
            f'<MUSICAL_KEY VALUE="{_nml_key(t.key)}"/>'
            f'</ENTRY>')
    pl = "".join(f'<ENTRY><PRIMARYKEY TYPE="TRACK" KEY={quoteattr(k)}/></ENTRY>' for k in keys)
    xml = ('<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\n'
           '<NML VERSION="19"><HEAD COMPANY="www.native-instruments.com" PROGRAM="Traktor"/>'
           f'<COLLECTION ENTRIES="{len(entries)}">{"".join(entries)}</COLLECTION>'
           '<PLAYLISTS><NODE TYPE="FOLDER" NAME="$ROOT"><SUBNODES COUNT="1">'
           f'<NODE TYPE="PLAYLIST" NAME={quoteattr(name)}>'
           f'<PLAYLIST ENTRIES="{len(keys)}" TYPE="LIST" UUID="">{pl}</PLAYLIST></NODE>'
           '</SUBNODES></NODE></PLAYLISTS></NML>\n')
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)


def export(lib: Library, seq: list[int], path: str, name: str = "Tempest setlist"):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".m3u", ".m3u8"):
        write_m3u8(lib, seq, path, name)
    elif ext == ".csv":
        write_csv(lib, seq, path)
    elif ext == ".xml":
        write_rekordbox_xml(lib, {name: seq}, path)
    elif ext == ".nml":
        write_traktor_nml(lib, seq, path, name)
    else:
        raise ValueError(f"unsupported export format: {ext}")
