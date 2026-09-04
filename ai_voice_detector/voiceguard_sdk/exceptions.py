"""
Custom exceptions for the VoiceGuard SDK. Every public VoiceGuardClient
method raises one of these on failure -- callers never see a raw
grpc.RpcError or requests exception, just a clear, SDK-specific error
they can catch without importing gRPC or requests themselves.

Note: ConnectionError here intentionally shares its name with Python's
builtin ConnectionError (this is what the SDK spec calls for). Within
this package it's used unambiguously via `from .exceptions import
ConnectionError`; code that also needs the builtin in the same module
should import this one under an alias (e.g. `as VoiceGuardConnectionError`)
to avoid shadowing it.
"""


class VoiceGuardException(Exception):
    """Base class for every exception this SDK raises."""


class ConnectionError(VoiceGuardException):
    """The VoiceGuard server (gRPC or REST) could not be reached."""


class AudioFormatError(VoiceGuardException):
    """The given audio file could not be read or decoded."""


class SpeakerNotFoundError(VoiceGuardException):
    """The given speaker_id has never been enrolled."""
