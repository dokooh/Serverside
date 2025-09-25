#!/usr/bin/env python3
"""
Simple logger configuration without emoji for Windows compatibility
"""
import logging
import sys


def configure_simple_logger():
    """Configure logger without emojis for Windows compatibility"""
    # Create a custom formatter without emojis
    class SimpleFormatter(logging.Formatter):
        def __init__(self):
            super().__init__(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        def format(self, record):
            # Remove common emoji characters from the message
            if hasattr(record, 'msg') and record.msg:
                # Common emojis used in the codebase
                emoji_map = {
                    '🚀': '[START]',
                    '🔧': '[DEBUG]',
                    '📁': '[FILE]',
                    '⬇️': '[DOWNLOAD]',
                    '🧪': '[TEST]',
                    '✅': '[SUCCESS]',
                    '❌': '[ERROR]',
                    '📊': '[REPORT]',
                    '🎉': '[COMPLETE]',
                    '🔍': '[SEARCH]',
                    '💡': '[INFO]',
                    '⚠️': '[WARNING]',
                    '🏁': '[FINISH]'
                }
                
                message = str(record.msg)
                for emoji, replacement in emoji_map.items():
                    message = message.replace(emoji, replacement)
                record.msg = message
            
            return super().format(record)
    
    # Only configure if we haven't already
    root_logger = logging.getLogger()
    if not any(isinstance(h.formatter, SimpleFormatter) for h in root_logger.handlers):
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create new handler with simple formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(SimpleFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


if __name__ == "__main__":
    configure_simple_logger()
    logging.info("✅ Simple logger configured")