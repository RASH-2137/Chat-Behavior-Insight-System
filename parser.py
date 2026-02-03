"""
WhatsApp chat parser module
Converts raw WhatsApp chat export to structured DataFrame
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Union, TextIO, BinaryIO
from io import TextIOWrapper


def parse_whatsapp_chat(file_path: Union[str, TextIO, BinaryIO]) -> pd.DataFrame:
    """
    Parse WhatsApp chat export file into DataFrame
    
    Supports multiple date formats:
    - DD/MM/YYYY with 12-hour format (AM/PM)
    - MM/DD/YYYY with 12-hour format (AM/PM) 
    - YYYY-MM-DD with 24-hour format
    
    Args:
        file_path: Path to WhatsApp chat export (.txt file) or file-like object
        
    Returns:
        DataFrame with columns: timestamp, user, message
    """
    messages = []
    
    # Handle both file paths and file-like objects (like from Streamlit)
    if isinstance(file_path, str):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        # It's a file-like object
        try:
            if hasattr(file_path, 'seek'):
                file_path.seek(0)
            content = file_path.read()
            # Might be bytes, need to decode
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        except (AttributeError, UnicodeDecodeError):
            # Try wrapping it as text
            if hasattr(file_path, 'seek'):
                file_path.seek(0)
            content = TextIOWrapper(file_path, encoding='utf-8').read()
        
        # Reset in case it gets reused
        if hasattr(file_path, 'seek'):
            file_path.seek(0)
    
    # Comprehensive pattern that handles multiple date/time formats
    pattern = re.compile(
        r"""
        ^(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}),?\s*
        (\d{1,2}:\d{2}(?:\s?[AP]M)?)\s*-\s*
        (.*?):\s
        (.*)$
        """,
        re.VERBOSE | re.MULTILINE
    )
    
    matches = pattern.findall(content)
    
    for match in matches:
        date_str, time_str, user, message = match
        datetime_str = f"{date_str}, {time_str}".strip()
        
        # Try different date formats
        date_formats = [
            '%d/%m/%Y, %I:%M %p',  # DD/MM/YYYY with AM/PM
            '%d/%m/%y, %I:%M %p',  # DD/MM/YY with AM/PM
            '%m/%d/%Y, %I:%M %p',  # MM/DD/YYYY with AM/PM
            '%m/%d/%y, %I:%M %p',  # MM/DD/YY with AM/PM
            '%Y-%m-%d, %H:%M',      # YYYY-MM-DD 24-hour
            '%d/%m/%Y, %H:%M',      # DD/MM/YYYY 24-hour
            '%m/%d/%Y, %H:%M',      # MM/DD/YYYY 24-hour
        ]
        
        timestamp = None
        for fmt in date_formats:
            try:
                timestamp = pd.to_datetime(datetime_str, format=fmt)
                break
            except:
                continue
        
        # If format matching failed, try pandas flexible parser
        if timestamp is None:
            try:
                timestamp = pd.to_datetime(datetime_str, dayfirst=True, errors='coerce')
            except:
                continue
        
        if pd.notna(timestamp):
            messages.append({
                'timestamp': timestamp,
                'user': user.strip(),
                'message': message.strip()
            })
    
    df = pd.DataFrame(messages)
    
    if df.empty:
        raise ValueError(
            "No messages found in chat file. Please check the file format. "
            "Supported formats: DD/MM/YYYY or MM/DD/YYYY with AM/PM, or YYYY-MM-DD with 24-hour time."
        )
    
    return df
