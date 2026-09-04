# AI Voice Detector - SIH26104

Real-time detection of AI-generated/cloned voice for fraud prevention.

## Setup

```bash
# from the ai_voice_detector/ directory
python -m venv venv

# activate the virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Live Phone Call Monitoring (Twilio)

Analyzes an active phone call in real time via Twilio Media Streams: Twilio
streams the call's audio to the app over a websocket, which scores it in
rolling 2-second windows (same logic as the "Simulate Live Stream" demo) and
publishes the current risk to the dashboard.

**Test the pipeline first, without a real call**: see "Testing without a real
call" below. Confirm that end-to-end before wiring up an actual phone number.

**Prerequisites**: a Twilio account with a phone number provisioned, and
[ngrok](https://ngrok.com/download) installed to expose your local server.

1. Install ngrok, then expose your local Flask server (must be running on
   port 5000 first -- see step 4):
   ```bash
   ngrok http 5000
   ```
   Copy the `https://...ngrok...` URL it prints.

2. In the Twilio console, open your phone number's configuration and set:
   **Voice → A call comes in → Webhook** to
   `https://YOUR_NGROK_URL/incoming-call`, method **HTTP POST**.

3. Set the required environment variables (never hardcode these --
   `twilio_config.py` reads them from the environment only):
   ```bash
   export TWILIO_ACCOUNT_SID=your_account_sid
   export TWILIO_AUTH_TOKEN=your_auth_token
   export TWILIO_PHONE_NUMBER=your_twilio_number
   ```
   (PowerShell: `$env:TWILIO_ACCOUNT_SID = "..."`, etc.)

4. Run the app:
   ```bash
   python app.py
   ```

5. Call your Twilio number from any phone.

6. Watch the dashboard (`http://localhost:5000`) -- click **Start Monitoring**
   under "Live Phone Call" to see the rolling risk score update as the call
   progresses.

### Testing without a real call

`simulate_twilio_stream.py` replays a local audio file as if it were a live
Twilio Media Stream (same base64 mulaw-8kHz frames, same 20ms pacing), sent
directly to your locally running `/media-stream` websocket -- no Twilio
account, phone call, or ngrok tunnel needed for this step:

```bash
python app.py                                    # in one terminal
python simulate_twilio_stream.py test/pc_bonafide.wav   # in another
```

It prints the rolling score as each 2-second chunk is scored, just like a
real call would show on the dashboard.

## gRPC API

Alongside the REST API (port 5000), `app.py` also starts a gRPC server on
**port 50051** -- for enterprise integrations (e.g. a bank's core banking
system) that call this service via gRPC instead of HTTP/JSON. Both run in
the same process; starting `python app.py` starts both automatically.

The service definition lives in `proto/voiceguard.proto`. Audio is sent as
raw 32-bit float PCM bytes (mono) plus a separate sample rate field -- not
a WAV file -- matching the in-memory representation the Python pipeline
already uses internally.

Example: how a banking system would call `AnalyzeWithContext` in Python
(the same pattern applies to any language with a gRPC/protobuf codegen tool):

```python
import grpc
import numpy as np
import soundfile as sf

import voiceguard_pb2
import voiceguard_pb2_grpc

channel = grpc.insecure_channel("voiceguard.internal.bank.example:50051")
stub = voiceguard_pb2_grpc.VoiceGuardStub(channel)

audio, sr = sf.read("call_recording.wav", dtype="float32")

response = stub.AnalyzeWithContext(voiceguard_pb2.ContextualRequest(
    audio_data=audio.tobytes(),
    sample_rate=sr,
    profile="high_value_transaction",
    caller_known=True,
    channel="voip",
    transaction_amount=50000.0,
    new_beneficiary=True,
    speaker_id="customer_1234567890",
))

print(response.risk_level, response.final_risk, response.recommended_action)
```

To regenerate the Python stubs after editing the `.proto` file:

```bash
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/voiceguard.proto
```

To test the running server end-to-end without writing a client:

```bash
python app.py                  # in one terminal -- starts REST (5000) + gRPC (50051)
python grpc_client_test.py     # in another -- exercises all four RPCs
```

## Dataset

_TODO: describe data sources, licensing, and how real/ and fake/ samples are collected._

## Usage

_TODO: document how to run training, evaluation, and the detection app/API._

## Architecture

_TODO: describe the feature extraction pipeline, model, and system design._

## Results

_TODO: add evaluation metrics and findings._
