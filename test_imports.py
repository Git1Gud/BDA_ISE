#!/usr/bin/env python3
"""Test script for modular imports."""

try:
    from src import StudyMaterialRAG, RAGConfig
    print("SUCCESS: All imports working")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
