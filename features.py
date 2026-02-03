"""
Feature engineering module
Extracts behavioral features from chat messages
"""

import pandas as pd
import numpy as np
from typing import Dict
import re


def calculate_message_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate message length stats per user"""
    user_stats = df.groupby('user')['message'].agg([
        ('avg_length', lambda x: x.str.len().mean()),
        ('median_length', lambda x: x.str.len().median()),
        ('total_chars', lambda x: x.str.len().sum())
    ]).reset_index()
    
    return user_stats


def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate temporal activity features per user"""
    df = df.copy()  # Don't mess with the original
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Calculate messages per day for each user
    df['date'] = df['timestamp'].dt.date
    messages_per_day = df.groupby(['user', 'date']).size().reset_index(name='daily_count')
    avg_messages_per_day = messages_per_day.groupby('user')['daily_count'].mean().reset_index(name='messages_per_day')
    
    # Response time - time between a user's consecutive messages
    df_sorted = df.sort_values('timestamp')
    df_sorted['time_diff'] = df_sorted.groupby('user')['timestamp'].diff()
    avg_response_time = df_sorted.groupby('user')['time_diff'].mean().dt.total_seconds() / 3600  # hours
    avg_response_time = avg_response_time.reset_index(name='avg_response_time_hours')
    avg_response_time['avg_response_time_hours'] = avg_response_time['avg_response_time_hours'].fillna(0)
    
    # Night activity - messages between 10 PM and 6 AM
    night_mask = (df['hour'] >= 22) | (df['hour'] < 6)
    night_activity = df[night_mask].groupby('user').size() / df.groupby('user').size()
    night_activity = night_activity.reset_index(name='night_activity_ratio')
    night_activity['night_activity_ratio'] = night_activity['night_activity_ratio'].fillna(0)
    
    # Merge everything together
    temporal_features = avg_messages_per_day.merge(
        avg_response_time, on='user', how='left'
    ).merge(
        night_activity, on='user', how='left'
    )
    
    return temporal_features


def calculate_emotional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate emotional expressiveness features"""
    df = df.copy()
    
    # Regex for emojis - covers most common ones
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    
    # Count various emotional indicators
    df['exclamation_count'] = df['message'].str.count('!')
    df['question_count'] = df['message'].str.count('\?')
    df['emoji_count'] = df['message'].apply(lambda x: len(emoji_pattern.findall(x)))
    df['uppercase_ratio'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    
    # Average these per user
    emotional_features = df.groupby('user').agg({
        'exclamation_count': 'mean',
        'question_count': 'mean',
        'emoji_count': 'mean',
        'uppercase_ratio': 'mean'
    }).reset_index()
    
    emotional_features.columns = ['user', 'avg_exclamations', 'avg_questions', 'avg_emojis', 'uppercase_ratio']
    
    return emotional_features


def calculate_link_sharing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate link/resource sharing features"""
    df = df.copy()
    
    # Simple URL pattern - catches http and https links
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    df['has_link'] = df['message'].apply(lambda x: bool(url_pattern.search(x)))
    link_sharing = df.groupby('user').agg({
        'has_link': ['sum', 'mean']
    }).reset_index()
    link_sharing.columns = ['user', 'total_links', 'link_sharing_ratio']
    
    return link_sharing


def calculate_conversation_initiation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate conversation initiation features"""
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # A conversation starts if there's a gap > 1 hour (or it's the first message)
    df_sorted['time_diff'] = df_sorted['timestamp'].diff()
    df_sorted['is_conversation_start'] = (df_sorted['time_diff'] > pd.Timedelta(hours=1)) | (df_sorted.index == 0)
    
    # Count initiations per user
    initiations = df_sorted[df_sorted['is_conversation_start']].groupby('user').size().reset_index(name='initiations')
    
    # Total messages per user
    total_messages = df.groupby('user').size().reset_index(name='total_messages')
    
    # Calculate the ratio
    initiation_features = total_messages.merge(initiations, on='user', how='left')
    initiation_features['initiations'] = initiation_features['initiations'].fillna(0)
    initiation_features['initiation_ratio'] = initiation_features['initiations'] / initiation_features['total_messages']
    
    return initiation_features


def extract_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all behavioral features for each user
    
    Args:
        df: DataFrame with columns: timestamp, user, message
        
    Returns:
        DataFrame with user features
    """
    # Calculate all feature groups
    length_features = calculate_message_length_features(df)
    temporal_features = calculate_temporal_features(df)
    emotional_features = calculate_emotional_features(df)
    link_features = calculate_link_sharing_features(df)
    initiation_features = calculate_conversation_initiation_features(df)
    
    # Merge everything together
    features = length_features.merge(temporal_features, on='user', how='outer')
    features = features.merge(emotional_features, on='user', how='outer')
    features = features.merge(link_features, on='user', how='outer')
    features = features.merge(initiation_features, on='user', how='outer')
    
    # Fill missing values with 0
    features = features.fillna(0)
    
    return features
