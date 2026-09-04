"""
Packaging for the VoiceGuard SDK (voiceguard_sdk/). Note: the SDK's gRPC
path imports the top-level voiceguard_pb2 / voiceguard_pb2_grpc modules
generated from proto/voiceguard.proto (see grpc_server.py) -- for an
editable install (`pip install -e .`) run from this project root, those
stay importable since the root is on sys.path. A fully standalone
distribution would need those generated modules packaged alongside
voiceguard_sdk/ too; not needed for this project's own use.
"""
from setuptools import find_packages, setup

setup(
    name="voiceguard-sdk",
    version="0.1.0",
    description="Python SDK for the VoiceGuard voice-clone/impersonation detection engine.",
    long_description=open("voiceguard_sdk/README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["voiceguard_sdk", "voiceguard_sdk.*"]),
    install_requires=[
        "grpcio>=1.60",
        "requests>=2.28",
        "numpy>=1.24",
    ],
    entry_points={
        "console_scripts": [
            "voiceguard=voiceguard_sdk.cli:main",
        ],
    },
    python_requires=">=3.9",
)
