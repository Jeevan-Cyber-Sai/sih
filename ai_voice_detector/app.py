"""
Flask web API + dashboard for the AI voice detector.
"""
import json
import os
import tempfile
import threading

import librosa
from flask import Flask, Response, jsonify, render_template, request
from flask_sock import Sock

import grpc_server
from alerting import send_test_alert, send_test_sms
from predict import analyze_file, analyze_file_with_context, stream_chunks
from risk_engine import list_profiles, load_config
from speaker_consistency import enroll_speaker, get_enrolled_speakers, verify_speaker
from telnyx_handler import init_telnyx
from twilio_handler import get_live_call_state, init_twilio

app = Flask(__name__)
# One Sock instance shared across providers -- flask-sock isn't meant to
# be attached to the same Flask app twice.
_sock = Sock(app)
init_twilio(app, _sock)
init_telnyx(app, _sock)

ALLOWED_EXT = {".wav", ".flac", ".mp3", ".ogg"}


def save_upload(file_storage):
    ext = os.path.splitext(file_storage.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        ext = ".wav"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    file_storage.save(path)
    return path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400

    path = save_upload(request.files["audio"])
    try:
        result = analyze_file(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    return jsonify(result)


@app.route("/analyze_stream", methods=["POST"])
def analyze_stream_route():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400

    path = save_upload(request.files["audio"])

    def event_stream():
        try:
            for result in stream_chunks(path):
                yield f"data: {json.dumps(result)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/live-call-status", methods=["GET"])
def live_call_status():
    return jsonify(get_live_call_state())


@app.route("/profiles", methods=["GET"])
def profiles():
    config = load_config()
    return jsonify({
        "default_profile": config.get("default_profile", "routine"),
        "profiles": config["profiles"],
    })


def _parse_bool_field(form, key):
    val = form.get(key)
    return val is not None and val.lower() in ("true", "on", "1", "yes")


def _parse_context_from_form(form):
    context = {
        "caller_known": _parse_bool_field(form, "caller_known"),
        "origin": form.get("origin") or None,
        "channel": form.get("channel") or None,
        "new_beneficiary": _parse_bool_field(form, "new_beneficiary"),
        "outside_business_hours": _parse_bool_field(form, "outside_business_hours"),
        "previously_flagged": _parse_bool_field(form, "previously_flagged"),
    }
    amount_raw = form.get("transaction_amount")
    try:
        context["transaction_amount"] = float(amount_raw) if amount_raw else 0.0
    except ValueError:
        context["transaction_amount"] = 0.0
    return context


@app.route("/analyze_with_context", methods=["POST"])
def analyze_with_context_route():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400

    profile = request.form.get("profile") or None
    context = _parse_context_from_form(request.form)
    speaker_id = request.form.get("speaker_id") or None

    path = save_upload(request.files["audio"])
    try:
        result = analyze_file_with_context(path, context, profile, speaker_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    return jsonify(result)


@app.route("/enroll-speaker", methods=["POST"])
def enroll_speaker_route():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400
    speaker_id = request.form.get("speaker_id")
    if not speaker_id:
        return jsonify({"error": "speaker_id is required"}), 400

    path = save_upload(request.files["audio"])
    try:
        audio, _ = librosa.load(path, sr=16000, mono=True)
        enroll_speaker(speaker_id, audio, sr=16000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    return jsonify({"speaker_id": speaker_id, "enrolled": True})


@app.route("/verify-consistency", methods=["POST"])
def verify_consistency_route():
    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"error": "no audio file uploaded (field name 'audio')"}), 400
    speaker_id = request.form.get("speaker_id")
    if not speaker_id:
        return jsonify({"error": "speaker_id is required"}), 400

    path = save_upload(request.files["audio"])
    try:
        audio, _ = librosa.load(path, sr=16000, mono=True)
        result = verify_speaker(speaker_id, audio, sr=16000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    if result is None:
        return jsonify({"error": f"speaker_id '{speaker_id}' is not enrolled", "speaker_id": speaker_id}), 404

    result["speaker_id"] = speaker_id
    return jsonify(result)


@app.route("/enrolled-speakers", methods=["GET"])
def enrolled_speakers_route():
    return jsonify({"speakers": get_enrolled_speakers()})


@app.route("/test-alert", methods=["POST"])
def test_alert_route():
    result = send_test_alert()
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/test-sms", methods=["POST"])
def test_sms_route():
    result = send_test_sms()
    return jsonify(result), (200 if result.get("ok") else 400)


@app.route("/alert-config", methods=["GET"])
def alert_config_route():
    config = load_config().get("alerting", {})
    # Never return credentials -- just enough for the dashboard's status
    # indicator to show whether alerting is on and roughly configured.
    return jsonify({
        "email_enabled": config.get("email_enabled", False),
        "recipient_count": len(config.get("alert_recipients", [])),
        "alert_on": config.get("alert_on", []),
        "sms_enabled": config.get("sms_enabled", False),
        "sms_recipient_count": len(config.get("sms_recipients", [])),
        "sms_alert_on": config.get("sms_alert_on", []),
    })


if __name__ == "__main__":
    # Flask's debug reloader re-executes this whole module in a child
    # process; without this guard, both the launcher and the real worker
    # would try to bind the gRPC port and the second one would crash.
    # WERKZEUG_RUN_MAIN is only set in the actual worker process.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=grpc_server.serve, kwargs={"block": True}, daemon=True).start()

    # threaded=True: a live call's /media-stream websocket, dashboard
    # polling of /live-call-status, and any file-upload requests all need
    # to be served concurrently, not one-at-a-time.
    app.run(debug=True, port=5000, threaded=True)
