"""
Telnyx Call Control integration: the Telnyx equivalent of
twilio_handler.py. Architecturally different from Twilio -- Telnyx is
webhook + REST-command driven (a webhook tells you a call happened, you
respond by issuing separate REST API calls) rather than "return markup
telling the provider what to do next". Once audio is flowing, though,
the actual decode/buffer/score/backpressure logic is identical, so this
reuses twilio_handler.CallSession as-is: Telnyx's default one-way stream
format is PCMU (G.711 mu-law) at 8kHz, base64-encoded over the
websocket -- the exact same wire format Twilio uses, decoded the exact
same way.

Flow: Telnyx POSTs a "call.initiated" webhook when your number rings ->
we call the Answer action -> Telnyx POSTs "call.answered" -> we call the
Start Streaming action pointing at our websocket -> audio flows in as
base64 mulaw media events, same shape as Twilio's.

Two-party flow (caller <-> a second real phone, with the caller's voice
monitored throughout): when the INBOUND leg answers, we both (a) start
streaming that leg's audio to us, and (b) Dial a second, outbound leg to
a verified number. When THAT second leg answers, we Bridge it to the
first leg so the two parties can actually talk. The two legs are
correlated via Telnyx's `client_state` (an opaque, base64-encoded string
Telnyx echoes back on every webhook for a call) -- we stash the original
inbound call_control_id there when placing the outbound Dial, so the
outbound leg's own call.answered webhook tells us exactly which inbound
leg to bridge it with, with no extra server-side state needed.

NOTE: some exact field/event names below (particularly in the
streaming_start/dial/bridge request bodies and the websocket's own event
names) are based on Telnyx's public docs, not hands-on verification --
Telnyx's portal has a "Debugging" section that shows real webhook
payloads and API responses, which is the fastest way to correct anything
that turns out to be named slightly differently once a real call is
tested.
"""
import base64
import json
import logging

import requests
from flask import Response, request
from flask_sock import Sock

from telnyx_config import (
    TELNYX_API_KEY,
    TELNYX_BRIDGE_TO_NUMBER,
    TELNYX_CONNECTION_ID,
    TELNYX_FROM_NUMBER,
    validate_bridge_config,
    validate_config,
)
from twilio_handler import CallSession, live_call_state, _state_lock

logger = logging.getLogger(__name__)

TELNYX_API_BASE = "https://api.telnyx.com/v2"


def _telnyx_post(path, json_body=None):
    validate_config()
    headers = {"Authorization": f"Bearer {TELNYX_API_KEY}"}
    r = requests.post(f"{TELNYX_API_BASE}{path}", headers=headers, json=json_body or {}, timeout=10)
    if r.status_code >= 300:
        logger.error("Telnyx API call %s failed: status=%s body=%s", path, r.status_code, r.text)
    return r


def answer_call(call_control_id):
    _telnyx_post(f"/calls/{call_control_id}/actions/answer")


def start_streaming(call_control_id, stream_url):
    _telnyx_post(f"/calls/{call_control_id}/actions/streaming_start", {
        "stream_url": stream_url,
        "stream_track": "inbound_track",
    })


def _encode_client_state(inbound_call_control_id):
    return base64.b64encode(inbound_call_control_id.encode()).decode()


def _decode_client_state(client_state):
    if not client_state:
        return None
    try:
        return base64.b64decode(client_state).decode()
    except Exception:
        return None


def dial_second_leg(inbound_call_control_id):
    """Places the outbound leg to the verified bridge-to number, tagging
    it with the inbound leg's call_control_id (via client_state) so its
    own call.answered webhook knows what to bridge with."""
    validate_bridge_config()
    _telnyx_post("/calls", {
        "connection_id": TELNYX_CONNECTION_ID,
        "to": TELNYX_BRIDGE_TO_NUMBER,
        "from": TELNYX_FROM_NUMBER,
        "client_state": _encode_client_state(inbound_call_control_id),
    })


def bridge_calls(this_call_control_id, other_call_control_id):
    _telnyx_post(f"/calls/{this_call_control_id}/actions/bridge", {
        "call_control_id": other_call_control_id,
    })


def telnyx_webhook():
    body = request.get_json(force=True, silent=True) or {}
    data = body.get("data", {})
    event_type = data.get("event_type")
    payload = data.get("payload", {})
    call_control_id = payload.get("call_control_id")
    client_state = _decode_client_state(payload.get("client_state"))

    logger.info(
        "Telnyx webhook: event_type=%s call_control_id=%s client_state=%s",
        event_type, call_control_id, client_state,
    )

    try:
        if event_type == "call.initiated" and call_control_id and client_state is None:
            # A fresh inbound call from a real caller (never true for our
            # own outbound leg, since we always set client_state on that
            # Dial request) -- answer it.
            answer_call(call_control_id)

        elif event_type == "call.answered" and call_control_id:
            if client_state is None:
                # The inbound caller leg just answered: start monitoring
                # their audio, and place the second, outbound leg.
                stream_url = f"wss://{request.host}/telnyx-media"
                start_streaming(call_control_id, stream_url)
                dial_second_leg(call_control_id)
            else:
                # The outbound (bridge-to) leg just answered: client_state
                # carries the original inbound leg's call_control_id.
                bridge_calls(call_control_id, client_state)

        elif event_type in ("call.hangup", "streaming.stopped"):
            with _state_lock:
                live_call_state["active"] = False
    except Exception:
        logger.exception("telnyx_webhook: error handling event_type=%s", event_type)

    # Telnyx expects a fast 200 ack regardless -- the actual call control
    # happens via the REST calls above, not the webhook response body.
    return Response("", status=200)


def init_telnyx(app, sock=None):
    """Registers the /telnyx-incoming-call HTTP webhook and the
    /telnyx-media websocket route on an existing Flask app."""
    app.add_url_rule("/telnyx-incoming-call", view_func=telnyx_webhook, methods=["POST"])

    sock = sock or Sock(app)

    @sock.route("/telnyx-media")
    def telnyx_media(ws):
        session = CallSession()
        try:
            while True:
                message = ws.receive()
                if message is None:
                    break

                data = json.loads(message)
                event = data.get("event")

                if event in ("start", "connected"):
                    call_id = (
                        data.get("start", {}).get("call_control_id")
                        or data.get("stream_id")
                        or data.get("call_control_id")
                    )
                    session.on_start(call_id)
                elif event == "media":
                    session.on_media(data["media"]["payload"])
                elif event == "stop":
                    session.on_stop()
                    break
                # anything else (e.g. a media-format negotiation event) is
                # informational only -- nothing to do.
        except Exception:
            logger.exception("telnyx-media: websocket handler error (sid=%s)", session.call_sid)
            with _state_lock:
                live_call_state["active"] = False

    return app
