"""Recoverable failures in the local processing pipeline."""


class PipelineError(Exception):
    """The pipeline understood the request and could not complete it."""


class VideoReadError(PipelineError):
    """A video file could not be opened or sampled."""


class AudioReadError(PipelineError):
    """An audio file could not be decoded."""
