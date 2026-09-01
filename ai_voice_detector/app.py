"""
Flask web API + dashboard for the AI voice detector.
"""
import json
import os
import tempfile

from flask import Flask, Response, jsonify, render_template, request

from predict import analyze_file, stream_chunks

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
