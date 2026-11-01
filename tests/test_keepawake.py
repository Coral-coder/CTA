"""Keep playing with the lid closed: what is changed on each system, and that it is put back."""
import pytest

import cta.keepawake as K


class FakeSettings(dict):
    def value(self, k, d=None):
        return self.get(k, d)

    def setValue(self, k, v):
        self[k] = v

    def remove(self, k):
        self.pop(k, None)


class R:
    def __init__(self, rc=0, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


class P:
    def __init__(self, args):
        self.args, self.killed = args, False

    def poll(self):
        return None

    def kill(self):
        self.killed = True


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.delenv("CTA_NO_KEEPAWAKE", raising=False)
    calls, started = [], []
    monkeypatch.setattr(K, "_detach", lambda args: started.append(P(args)) or started[-1])
    return calls, started


def test_windows_sets_the_lid_to_do_nothing_and_puts_it_back(fake, monkeypatch):
    calls, started = fake
    lid = {"ac": 1, "dc": 1}

    def run(args, **kw):
        calls.append(args)
        if args[:2] == ["powercfg", "/q"]:
            return R(0, f"    Current AC Power Setting Index: 0x{lid['ac']:08x}\n"
                        f"    Current DC Power Setting Index: 0x{lid['dc']:08x}\n")
        if args[1] == "/setacvalueindex":
            lid["ac"] = int(args[-1])
        if args[1] == "/setdcvalueindex":
            lid["dc"] = int(args[-1])
        return R()
    monkeypatch.setattr(K, "_run", run)
    monkeypatch.setattr(K, "_win_exec_state", lambda on: calls.append(("exec", on)))
    monkeypatch.setattr(K.sys, "platform", "win32")
    st = FakeSettings()
    ka = K.KeepAwake(st)
    assert ka.on() and lid == {"ac": 0, "dc": 0} and st["lid_saved"] == "1,1"
    assert ("exec", True) in calls
    assert any("Wait-Process" in " ".join(p.args) for p in started)     # the watcher puts it back after a crash
    lid["ac"] = 1                                                       # something turned it back on
    ka.reassert()
    assert lid["ac"] == 0
    ka.off()
    assert lid == {"ac": 1, "dc": 1} and "lid_saved" not in st and ("exec", False) in calls
    assert all(p.killed for p in started)


def test_a_live_update_hands_the_lid_over(fake, monkeypatch):
    calls, started = fake
    lid = {"ac": 0, "dc": 0}                                            # the old version set "do nothing"

    def run(args, **kw):
        if args[:2] == ["powercfg", "/q"]:
            return R(0, f": 0x{lid['ac']:08x}\n: 0x{lid['dc']:08x}\n")
        if args[1] == "/setacvalueindex":
            lid["ac"] = int(args[-1])
        if args[1] == "/setdcvalueindex":
            lid["dc"] = int(args[-1])
        return R()
    monkeypatch.setattr(K, "_run", run)
    monkeypatch.setattr(K, "_win_exec_state", lambda on: None)
    monkeypatch.setattr(K.sys, "platform", "win32")
    st = FakeSettings(lid_saved="1,2")                                  # ... and saved the DJ's original
    new = K.KeepAwake(st)
    new.on()
    old = K.KeepAwake(st)
    old.active = True
    old.off(restore=False)                                              # the old one leaves without resetting
    assert lid == {"ac": 0, "dc": 0}
    new.off()
    assert lid == {"ac": 1, "dc": 2}                                    # the original comes back at the end


def test_mac_disables_lid_sleep_only_after_setup(fake, monkeypatch):
    calls, started = fake
    ready = {"ok": False}

    def run(args, **kw):
        calls.append(args)
        if "-l" in args:
            return R(0 if ready["ok"] else 1)
        return R()
    monkeypatch.setattr(K, "_run", run)
    monkeypatch.setattr(K.sys, "platform", "darwin")
    ka = K.KeepAwake()
    assert ka.on() is False                                             # idle sleep only: setup needed for the lid
    assert any(p.args[0] == "/usr/bin/caffeinate" for p in started)
    assert not any("disablesleep" in a for c in calls if "-l" not in c for a in c)
    ka.off()
    ready["ok"] = True
    ka = K.KeepAwake()
    assert ka.on() is True
    assert ["/usr/bin/sudo", "-n", K.PMSET, "-a", "disablesleep", "1"] in calls
    assert any("disablesleep 0" in " ".join(p.args) for p in started)  # the watcher puts it back after a crash
    ka.off()
    assert ["/usr/bin/sudo", "-n", K.PMSET, "-a", "disablesleep", "0"] in calls


def test_mac_setup_writes_one_narrow_rule(monkeypatch):
    got = {}
    monkeypatch.setattr(K.sys, "platform", "darwin")
    monkeypatch.setattr(K, "_run", lambda args, **kw: got.setdefault("args", args) and R(1))
    ok, why = K.mac_lid_setup()
    script = got["args"][2]
    assert "with administrator privileges" in script and "visudo -cf" in script
    assert "NOPASSWD: /usr/bin/pmset -a disablesleep 0, /usr/bin/pmset -a disablesleep 1" in script
    assert not ok


def test_tests_never_touch_the_machine(monkeypatch):
    monkeypatch.setenv("CTA_NO_KEEPAWAKE", "1")
    monkeypatch.setattr(K, "_run", lambda *a, **k: pytest.fail("touched the machine"))
    monkeypatch.setattr(K, "_detach", lambda *a, **k: pytest.fail("touched the machine"))
    ka = K.KeepAwake()
    assert ka.on()
    ka.off()
