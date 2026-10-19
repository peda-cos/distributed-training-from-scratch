"""Loader package for the spec's ``code/main.py`` implementation.

A top-level package named ``code`` would shadow Python's stdlib ``code``
module (used by pdb and codeop). Tests and spawned workers therefore import
``data_parallel_ddp.main``, which executes the real implementation under this
non-conflicting module name.
"""
