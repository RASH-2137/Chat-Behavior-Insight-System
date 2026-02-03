"""
Behavior profiling module.
Generates natural language behavior profiles for users.
"""

import pandas as pd
import numpy as np
from typing import Dict


# Default cluster names - only works when n_clusters=5
# For other cluster counts, you'll get generic "Cluster X" names
DEFAULT_CLUSTER_NAMES = {
    0: "Regular Participants",
    1: "Active Conversationalists",
    2: "Information Broadcasters",
    3: "Silent Observers",
    4: "Night Owls"
}


def get_cluster_name(cluster_id: int, cluster_names: dict = None) -> str:
    """
    Get human-readable cluster name.
    
    Args:
        cluster_id: Numeric cluster ID
        cluster_names: Optional custom mapping of cluster_id -> name.
                      If None, uses DEFAULT_CLUSTER_NAMES for IDs 0-4,
                      otherwise returns "Cluster {cluster_id}"
    
    Returns:
        Human-readable cluster name
    """
    if cluster_names is not None:
        return cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    return DEFAULT_CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")


def calculate_influence_score(features: pd.Series) -> float:
    """
    Calculate influence score based on activity and engagement.
    
    Args:
        features: Series with user features
        
    Returns:
        Influence score between 0 and 1
    """
    # Get the key metrics
    messages_per_day = features.get('messages_per_day', 0)
    initiation_ratio = features.get('initiation_ratio', 0)
    avg_response_time = features.get('avg_response_time_hours', 24)
    
    # Activity score - normalize to 5 msg/day = max score
    activity_score = min(messages_per_day / 5.0, 1.0)
    
    # Engagement - faster responses = higher score
    response_score = 1.0 - min(avg_response_time / 24.0, 1.0)
    
    # Initiation score
    initiation_score = min(initiation_ratio * 10, 1.0)
    
    # Weighted combination
    influence = (activity_score * 0.4 + response_score * 0.3 + initiation_score * 0.3)
    
    return min(max(influence, 0.0), 1.0)


def generate_behavior_profile(features: pd.Series, cluster_id: int) -> str:
    """
    Generate natural language behavior profile for a user.
    
    Args:
        features: Series with user features
        cluster_id: Cluster assignment
        
    Returns:
        Behavior profile description
    """
    cluster_name = get_cluster_name(cluster_id)
    
    profile_parts = [f"This user belongs to the '{cluster_name}' group."]
    
    # Activity level
    messages_per_day = features.get('messages_per_day', 0)
    total_messages = features.get('total_messages', 0)
    
    if messages_per_day >= 4 or total_messages >= 500:
        activity_level = "highly active"
    elif messages_per_day >= 2 or total_messages >= 100:
        activity_level = "moderately active"
    else:
        activity_level = "low activity"
    
    profile_parts.append(f"This user is {activity_level},")
    
    # Message length preference
    avg_length = features.get('avg_length', 0)
    if avg_length >= 100:
        profile_parts.append("writes long, detailed messages,")
    else:
        profile_parts.append("prefers short messages,")
    
    # Emotional expressiveness
    avg_emojis = features.get('avg_emojis', 0)
    avg_exclamations = features.get('avg_exclamations', 0)
    uppercase_ratio = features.get('uppercase_ratio', 0)
    
    emotional_score = avg_emojis + avg_exclamations * 0.5 + uppercase_ratio * 10
    
    if emotional_score >= 2:
        profile_parts.append("emotionally expressive,")
    else:
        profile_parts.append("emotionally reserved,")
    
    # Response speed
    avg_response_time = features.get('avg_response_time_hours', 24)
    if avg_response_time < 2:
        profile_parts.append("responds quickly,")
    else:
        profile_parts.append("responds slowly,")
    
    # Link sharing
    link_sharing_ratio = features.get('link_sharing_ratio', 0)
    if link_sharing_ratio > 0.1:
        profile_parts.append("often shares links or resources,")
    
    # Night activity
    night_activity = features.get('night_activity_ratio', 0)
    if night_activity > 0.3:
        profile_parts.append("more active at night,")
    
    # Conversation initiation
    initiation_ratio = features.get('initiation_ratio', 0)
    influence_score = calculate_influence_score(features)
    
    if initiation_ratio > 0.1 or influence_score > 0.95:
        profile_parts.append("frequently initiates conversations and influences group flow.")
    else:
        # Remove trailing comma and add period
        if profile_parts[-1].endswith(','):
            profile_parts[-1] = profile_parts[-1].rstrip(',') + '.'
        else:
            profile_parts.append(".")
    
    # Clean up the profile
    profile = " ".join(profile_parts)
    profile = profile.replace(" ,", ",").replace("..", ".").replace(",.", ".")
    
    return profile


def generate_profiles(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate behavior profiles for all users.
    
    Args:
        features_df: DataFrame with user features and cluster assignments
        
    Returns:
        DataFrame with profiles and influence scores added
    """
    features_df = features_df.copy()
    
    # Calculate influence scores
    features_df['influence_score'] = features_df.apply(
        lambda row: calculate_influence_score(row), axis=1
    )
    
    # Generate profiles
    features_df['behavior_profile'] = features_df.apply(
        lambda row: generate_behavior_profile(row, row['cluster']), axis=1
    )
    
    return features_df
