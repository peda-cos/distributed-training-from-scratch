"""Loader package for the capstone's ``code/main.py`` implementation.

A top-level package named ``code`` would shadow Python's stdlib ``code``
module.  Tests and spawned workers therefore import
``end_to_end_distributed.main``, which executes the real implementation under
this non-conflicting module name.
"""
