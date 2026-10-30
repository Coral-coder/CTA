"""HTTPS for the packaged app. A frozen Python has no certificate store of
its own (the Mac build failed every GitHub check with CERTIFICATE_VERIFY_FAILED),
so requests verify against the certifi bundle shipped with CTA, with the
system's own store as a fallback."""

from __future__ import annotations

import ssl
import urllib.request

_ctx: ssl.SSLContext | None = None


def ssl_context() -> ssl.SSLContext:
    global _ctx
    if _ctx is None:
        ctx = ssl.create_default_context()
        try:
            import certifi

            ctx.load_verify_locations(cafile=certifi.where())
        except Exception:  # noqa: BLE001 - no bundle: the system store alone
            pass
        try:
            ctx.load_default_certs()
        except Exception:  # noqa: BLE001
            pass
        _ctx = ctx
    return _ctx


def urlopen(req, timeout: float = 30):
    return urllib.request.urlopen(req, timeout=timeout, context=ssl_context())
