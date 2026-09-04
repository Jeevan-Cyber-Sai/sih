"""
Twilio Media Streams integration: lets a live phone call be scored in
real time, using the exact same 2-second/50%-overlap/alpha=0.6 rolling
logic that predict.stream_chunks() already uses for simulated file
streaming -- the only thing that changes is where the audio comes from
(a live 8kHz mulaw websocket feed instead of a pre-loaded file).

Flow: Twilio POSTs to /incoming-call, gets back TwiML telling it to open
a Media Stream websocket to /media-stream, then pushes base64-encoded
8kHz mulaw audio frames over that socket in near-real-time.
"""
import audioop
import base64
import json
import logging
import threading

import librosa
import numpy as np
from flask import Response, request
from flask_sock import Sock

from predict import load_model, risk_level, score_single_clip
from privacy import zero_buffer
from twilio_config import validate_config

logger = logging.getLogger(__name__)

TWILIO_SR = 8000
TARGET_SR = 16000
CHUNK_SECONDS = 2.0
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)  # 32000
ALPHA = 0.6  # same smoothing factor as predict.stream_chunks()

# Shared with app.py's GET /live-call-status for dashboard polling.
# Coarse-grained lock: reads/writes are a handful of dict keys, not a
# hot path (one update per 1-second chunk, one poll per second).
_state_lock = threading.Lock()
live_call_state = {
    "active": False,
    "call_sid": None,
    "rolling_score": None,
    "risk_level": None,
    "recommended_action": None,
    "chunk_count": 0,
}


def get_live_call_state():
    with _state_lock:
        return dict(live_call_state)


def _reset_state(call_sid=None, active=True):
    with _state_lock:
        live_call_state.update({
            "active": active,
            "call_sid": call_sid,
            "rolling_score": None,
            "risk_level": None,
            "recommended_action": None,
            "chunk_count": 0,
        })


def _publish_result(rolling_score, level, action):
    with _state_lock:
        live_call_state["rolling_score"] = rolling_score
        live_call_state["risk_level"] = level
        live_call_state["recommended_action"] = action
        live_call_state["chunk_count"] += 1
        return live_call_state["chunk_count"]


class CallSession:
    """Per-connection state for one live Media Stream: decode buffer,
    rolling score, and the busy flag backing the backpressure guard."""

    def __init__(self):
        self.buffer = np.zeros(0, dtype=np.float32)
        self.rolling_score = None
        self.busy = False
        self.call_sid = None
        self.clf, self.scaler = load_model()

    def on_start(self, call_sid):
        self.call_sid = call_sid
        _reset_state(call_sid=call_sid, active=True)
        logger.info("media-stream: call started (sid=%s)", call_sid)

    def on_media(self, payload_b64):
        mulaw_bytes = base64.b64decode(payload_b64)
        pcm16_bytes = audioop.ulaw2lin(mulaw_bytes, 2)  # 16-bit linear PCM, still 8kHz
        pcm = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        resampled = librosa.resample(pcm, orig_sr=TWILIO_SR, target_sr=TARGET_SR).astype(np.float32)
        self.buffer = np.concatenate([self.buffer, resampled])

        if len(self.buffer) < CHUNK_SAMPLES:
            return

        if self.busy:
            # Backpressure guard: never queue a backlog of chunks -- that
            # compounds lag until the live score is arbitrarily stale.
            # Drop this ready chunk and keep listening instead.
            logger.warning(
                "media-stream: dropping ready chunk (sid=%s) -- previous chunk still scoring",
                self.call_sid,
            )
            self.buffer = np.zeros(0, dtype=np.float32)
            return

        chunk = self.buffer[:CHUNK_SAMPLES].copy()
        self.buffer = self.buffer[CHUNK_SAMPLES // 2:]  # 50% overlap retained
        self.busy = True
        threading.Thread(target=self._score_chunk, args=(chunk,), daemon=True).start()

    def _score_chunk(self, chunk):
        try:
            instant_score = score_single_clip(chunk, self.clf, self.scaler)
            zero_buffer(chunk)

            self.rolling_score = instant_score if self.rolling_score is None else (
                ALPHA * instant_score + (1 - ALPHA) * self.rolling_score
            )
            self.rolling_score = round(self.rolling_score, 2)
            level, action = risk_level(self.rolling_score)

            chunk_count = _publish_result(self.rolling_score, level, action)
            logger.info(
                "media-stream chunk %d (sid=%s): instant=%.2f rolling=%.2f risk=%s",
                chunk_count, self.call_sid, instant_score, self.rolling_score, level,
            )
        except Exception:
            logger.exception("media-stream: scoring failed (sid=%s)", self.call_sid)
        finally:
            self.busy = False

    def on_stop(self):
        logger.info(
            "media-stream: call stopped (sid=%s) -- final rolling score=%s",
            self.call_sid, self.rolling_score,
        )
        with _state_lock:
            live_call_state["active"] = False
        self.buffer = np.zeros(0, dtype=np.float32)
        self.rolling_score = None


def incoming_call():
    """TwiML instructing Twilio to open a Media Stream to /media-stream
    on this same host (works behind ngrok: request.host reflects
    whatever public hostname Twilio actually connected to).

    Credentials are validated here rather than at app startup, so
    scripts/simulate_twilio_stream.py can exercise /media-stream directly
    (bypassing Twilio and this route entirely) without requiring real
    Twilio credentials to be configured -- a real call, which does hit
    this route, still fails loudly and immediately if misconfigured."""
    validate_config()
    ws_url = f"wss://{request.host}/media-stream"
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect><Stream url=\"" + ws_url + "\" /></Connect></Response>"
    )
    return Response(twiml, mimetype="text/xml")


def init_twilio(app, sock=None):
    """Registers the /incoming-call HTTP route and the /media-stream
    websocket route on an existing Flask app. `sock` lets multiple
    telephony providers (see telnyx_handler.py) share one Sock instance
    on the same app -- flask-sock isn't meant to be attached twice."""
    app.add_url_rule("/incoming-call", view_func=incoming_call, methods=["POST"])

    sock = sock or Sock(app)

    @sock.route("/media-stream")
    def media_stream(ws):
        session = CallSession()
        try:
            while True:
                message = ws.receive()
                if message is None:
                    break

                data = json.loads(message)
                event = data.get("event")

                if event == "start":
                    session.on_start(data.get("start", {}).get("callSid"))
                elif event == "media":
                    session.on_media(data["media"]["payload"])
                elif event == "stop":
                    session.on_stop()
                    break
                # "connected" and any other event types are informational
                # only -- nothing to do.
        except Exception:
            logger.exception("media-stream: websocket handler error (sid=%s)", session.call_sid)
            with _state_lock:
                live_call_state["active"] = False

    return app
