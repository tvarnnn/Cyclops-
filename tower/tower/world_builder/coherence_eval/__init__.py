"""Coherence forensics and evaluation for World Builder reconstructions.

Diagnostic code, not a builder stage: everything here reads a persisted
(ideally frozen) world and measures it. Nothing here is imported by the
live builder path, and nothing here may feed a forensic annotation --
region labels in particular -- back into reconstruction.
"""
