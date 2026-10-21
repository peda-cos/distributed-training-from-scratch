"""Loader package for the spec's ``code/main.py`` implementation.

A top-level package named ``code`` would shadow Python's stdlib ``code``
module. Tests and spawned workers therefore import ``zero_sharding.main``,
which executes the real implementation under this non-conflicting module name.
"""
