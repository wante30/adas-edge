#!/usr/bin/env python3
"""Deprecated entry point.

The single BAR1 control worker is now pcie_protocol_service.py.  Keeping this
name as a thin launcher prevents an older second writer from being started
alongside the unified service.
"""
from pcie_protocol_service import main

if __name__ == "__main__":
    raise SystemExit(main())
