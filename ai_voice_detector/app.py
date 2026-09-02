"""
Flask web API + dashboard for the AI voice detector.
"""
import json
import os
import tempfile

from flask import Flask, Response, jsonify, render_template, request

from predict import analyze_file, analyze_file_with_context, stream_chunks
from risk_engine import list_profiles, load_config

app = Flask(__name__)

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

    path = save_upload(request.files["audio"])
    try:
        result = analyze_file_with_context(path, context, profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
