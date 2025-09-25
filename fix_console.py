#!/usr/bin/env python3
"""
Fix console encoding issues for Windows
"""
import os
import sys

def setup_console_encoding():
    """Set up proper console encoding for emoji/unicode support"""
    if sys.platform == 'win32':
        # Set console to UTF-8 on Windows
        os.system('chcp 65001 > nul')
        
        # Set environment variables for better unicode support
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Try to set stdout/stderr encoding
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 doesn't have reconfigure
            pass
    
    return True

if __name__ == "__main__":
    setup_console_encoding()
    print("✅ Console encoding configured for unicode support")