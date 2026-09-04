"""
HTTP-range-backed file-like object that lets Python's zipfile module read
entries out of a huge remote .zip without downloading the whole archive.

Used to pull a labeled subset of ASVspoof2019 LA out of the ~7.1GB LA.zip
hosted on datashare.ed.ac.uk, whose connection resets roughly half the time,
so every network call here is retried.
"""
import time
import requests


class HTTPRangeFile:
    def __init__(self, url, session=None, max_retries=25, retry_wait=1.0):
        self.url = url
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.pos = 0
        self._cache_start = None
        self._cache_bytes = None
        self.length = self._fetch_length()

    def _fetch_length(self):
        last_exc = None
        for _ in range(self.max_retries):
            try:
                r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=30)
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    return int(cr.split("/")[-1])
                if "Content-Length" in r.headers and r.status_code == 200:
                    return int(r.headers["Content-Length"])
                # Got a response, but not one we can read a length from --
                # capture WHY instead of silently retrying and eventually
                # raising "None" with no diagnostic info at all.
                last_exc = RuntimeError(
                    f"status={r.status_code} headers={dict(r.headers)} "
                    f"body[:200]={r.text[:200]!r}"
                )
            except Exception as e:
                last_exc = e
            time.sleep(self.retry_wait)
        raise RuntimeError(f"could not determine remote file length: {last_exc}")

    def _ranged_get(self, start, end):
        """Fetch bytes [start, end] inclusive, with retries."""
        end = min(end, self.length - 1)
        if end < start:
            return b""
        want = end - start + 1
        headers = {"Range": f"bytes={start}-{end}"}
        last_exc = None
        for _ in range(self.max_retries):
            try:
                r = self.session.get(self.url, headers=headers, timeout=60)
                if r.status_code in (200, 206) and len(r.content) == want:
                    return r.content
                last_exc = RuntimeError(f"unexpected status {r.status_code} len {len(r.content)} want {want}")
            except Exception as e:
                last_exc = e
            time.sleep(self.retry_wait)
        raise RuntimeError(f"ranged GET failed for bytes {start}-{end}: {last_exc}")

    def prefetch(self, start, end):
        """Pull a byte range into the local cache so later read()/seek() calls
        in that span are served without hitting the network again."""
        self._cache_start = start
        self._cache_bytes = self._ranged_get(start, end)

    def read(self, n=-1):
        if n is None or n < 0:
            end = self.length - 1
        else:
            end = min(self.pos + n, self.length) - 1
        if end < self.pos:
            return b""

        if self._cache_start is not None:
            cache_end = self._cache_start + len(self._cache_bytes) - 1
            if self._cache_start <= self.pos and end <= cache_end:
                data = self._cache_bytes[self.pos - self._cache_start: end - self._cache_start + 1]
                self.pos += len(data)
                return data

        data = self._ranged_get(self.pos, end)
        self.pos += len(data)
        return data

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.length + offset
        else:
            raise ValueError("bad whence")
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True
