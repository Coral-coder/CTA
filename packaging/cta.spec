# PyInstaller spec - builds a self-contained folder (onedir: fast start-up,
# no temp-dir extraction, antivirus friendly).
#   Windows: Tempest.exe (GUI) + tempest-cli.exe (console) side by side
#   macOS:   Tempest.app (the app binary also accepts CLI commands)
#   Linux:   Tempest + tempest-cli
import os
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
SRC = os.path.join(ROOT, "src")
ENTRY = os.path.join(ROOT, "packaging", "cta_main.py")
MODEL = os.path.join(ROOT, "build_models", "discogs-effnet-bsdynamic-1.onnx")

sys.path.insert(0, SRC)
from cta import BUILD, VERSION, __version__  # noqa: E402

datas = [(os.path.join(SRC, "cta", "discogs400_labels.txt"), "cta"), (os.path.join(SRC, "cta", "icon.png"), "cta"),
         (os.path.join(SRC, "cta", "app-icon-180.png"), "cta"), (os.path.join(SRC, "cta", "app-icon-512.png"), "cta")]
if os.path.exists(MODEL):
    datas.append((MODEL, "models"))
    datas.append((os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), "."))
else:
    print("WARNING: AI model not bundled; it will be downloaded on first run")

binaries = collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("pylibrb")
hiddenimports = ["certifi", "cryptography"] + (
    collect_submodules("cta")
    + collect_submodules("av")
    + collect_submodules("onnxruntime.capi")
    + collect_submodules("pylibrb")
    + collect_submodules("rtmidi")
    + collect_submodules("qrcode")
    + ["PySide6.QtNetwork", "mutagen", "platformdirs", "sounddevice", "_sounddevice", "scipy.signal", "scipy.signal._sosfilt"]
)
excludes = [
    "tkinter", "yaml", "matplotlib", "pandas", "IPython", "pytest", "essentia",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "PySide6.QtPdf", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtBluetooth",
    "PySide6.QtPositioning", "PySide6.QtSensors", "PySide6.QtSerialPort",
]


def analysis():
    return Analysis(
        [ENTRY],
        pathex=[SRC],
        binaries=binaries,
        datas=datas,
        hiddenimports=hiddenimports,
        excludes=excludes,
        noarchive=False,
        optimize=1,
    )


a = analysis()
pyz = PYZ(a.pure)

icon = os.path.join(ROOT, "packaging", "icon.ico" if sys.platform == "win32" else "icon.icns")
icon = icon if os.path.exists(icon) else None

gui = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Tempest",
    console=False,
    disable_windowed_traceback=False,
    upx=False,
    icon=icon,
    argv_emulation=False,
)

exes = [gui]
if sys.platform != "darwin":
    cli = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="tempest-cli",
        console=True,
        upx=False,
        icon=icon,
    )
    exes.append(cli)

coll = COLLECT(*exes, a.binaries, a.datas, strip=False, upx=False, name="Tempest")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Tempest.app",
        icon=icon,
        bundle_identifier="app.tempest.dj",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": str(BUILD) if BUILD else VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSRequiresAquaSystemAppearance": False,
            # never "App Nap" a DJ app: a minimised / background Tempest keeps full
            # timing, so the audio never glitches
            "NSAppSleepDisabled": True,
            "LSAppNapIsDisabled": True,
        },
    )
