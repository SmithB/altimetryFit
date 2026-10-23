#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Set the thread-count environment variables for BLAS, OpenMP, and numexpr.

OpenBLAS and OpenMP read these variables only once, when they are first
loaded, so set_threads() has to run before numpy (or anything that imports
numpy) is imported.  altimetryFit/__init__.py calls it before its other
imports when running a command-line script that takes --THREADS.
"""

import os
import re
import sys


def set_threads(argv=None):
    """
    Set thread-count environment variables from a THREADS=N argument

    Parameters
    ----------
    argv : list of str, optional
        arguments to search for THREADS=N.  The default is sys.argv.

    Returns
    -------
    n_threads : str
        the number of threads (default "1", or $SLURM_NTASKS if set)
    """
    if argv is None:
        argv = sys.argv
    threads_re = re.compile(r"THREADS=(\S+)")
    n_threads = "1"
    for arg in argv:
        m = threads_re.search(arg)
        if m is not None:
            n_threads = m.group(1)
    if n_threads == "1" and "SLURM_NTASKS" in os.environ:
        n_threads = os.environ['SLURM_NTASKS']
        print(f"n_threads={n_threads}")
    os.environ["MKL_NUM_THREADS"] = n_threads
    os.environ["OPENBLAS_NUM_THREADS"] = n_threads
    os.environ['NUMEXPR_NUM_THREADS'] = n_threads
    os.environ['OMP_NUM_THREADS'] = n_threads
    return n_threads
