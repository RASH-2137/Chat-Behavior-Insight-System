"""
Main module for WhatsApp chat behavior analysis
"""

import pandas as pd
from typing import Tuple, Union, TextIO, BinaryIO
from parser import parse_whatsapp_chat
from features import extract_all_features
from clustering import assign_clusters
from profiling import generate_profiles, get_cluster_name
from report import generate_reports


def analyze_chat(file_path: str, n_clusters: int = 5, random_state: int = 42,
                 csv_output: str = None, pdf_output: str = None) -> pd.DataFrame:
    """
    Analyze WhatsApp chat and generate behavior reports
    
    Args:
        file_path: Path to WhatsApp chat export file (.txt)
        n_clusters: Number of clusters for KMeans (default: 5)
        random_state: Random state for reproducibility (default: 42)
        csv_output: Optional path for CSV report output
        pdf_output: Optional path for PDF report output
        
    Returns:
        DataFrame with user features, clusters, and behavior profiles
    """
    # Parse the chat file first
    print("Parsing WhatsApp chat file...")
    df = parse_whatsapp_chat(file_path)
    print(f"Parsed {len(df)} messages from {df['user'].nunique()} users")
    
    # Now extract all the features
    print("Extracting behavioral features...")
    features_df = extract_all_features(df)
    print(f"Extracted features for {len(features_df)} users")
    
    # Do the clustering
    print(f"Performing clustering with {n_clusters} clusters...")
    features_df = assign_clusters(features_df, n_clusters=n_clusters, random_state=random_state)
    print("Clustering complete")
    
    # Generate the behavior profiles
    print("Generating behavior profiles...")
    features_df = generate_profiles(features_df)
    print("Profiles generated")
    
    # Finally, generate the reports
    print("Generating reports...")
    generate_reports(features_df, csv_path=csv_output, pdf_path=pdf_output)
    print("Analysis complete!")
    
    return features_df


def analyze_chat_streamlit(file_input: Union[str, TextIO, BinaryIO], n_clusters: int = 5, 
                           random_state: int = 42, cluster_names: dict = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze WhatsApp chat from file-like input (for Streamlit/web apps)
    
    Doesn't write to filesystem - returns DataFrames directly. Good for 
    in-memory file uploads.
    
    Args:
        file_input: File path or file-like object (e.g., Streamlit uploader)
        n_clusters: Number of clusters (default: 5)
        random_state: Random state for reproducibility
        cluster_names: Optional dict mapping cluster_id -> name
        
    Returns:
        (final_report, cluster_summary) DataFrames
        
        final_report has: user, cluster, cluster_name, total_messages, 
        messages_per_day, influence_score, behavior_profile, plus other features
        
        cluster_summary has: cluster, cluster_name, user_count, 
        avg_messages_per_day, avg_influence_score
    """
    # Parse the uploaded file
    df = parse_whatsapp_chat(file_input)
    
    # Extract features
    features_df = extract_all_features(df)
    
    # Cluster users into groups
    features_df = assign_clusters(features_df, n_clusters=n_clusters, random_state=random_state)
    
    # Generate profiles for each user
    features_df = generate_profiles(features_df)
    
    # Build report - these are the main columns we want
    report_columns = [
        'user', 'cluster', 'total_messages', 'messages_per_day',
        'influence_score', 'behavior_profile'
    ]
    
    # Get what's available and add any other feature columns too
    available_report_cols = [col for col in report_columns if col in features_df.columns]
    additional_cols = [col for col in features_df.columns 
                      if col not in report_columns and col not in ['cluster', 'influence_score', 'behavior_profile']]
    
    final_report = features_df[available_report_cols + additional_cols].copy()
    
    # Add readable cluster names
    final_report['cluster_name'] = final_report['cluster'].apply(
        lambda x: get_cluster_name(x, cluster_names)
    )
    
    # Summary stats per cluster
    cluster_summary = features_df.groupby('cluster').agg({
        'user': 'count',
        'messages_per_day': 'mean',
        'influence_score': 'mean'
    }).reset_index()
    
    cluster_summary.columns = ['cluster', 'user_count', 'avg_messages_per_day', 'avg_influence_score']
    cluster_summary['cluster_name'] = cluster_summary['cluster'].apply(
        lambda x: get_cluster_name(x, cluster_names)
    )
    
    # Round to 2 decimals
    cluster_summary['avg_messages_per_day'] = cluster_summary['avg_messages_per_day'].round(2)
    cluster_summary['avg_influence_score'] = cluster_summary['avg_influence_score'].round(2)
    
    # Put columns in a nice order
    cluster_summary = cluster_summary[['cluster', 'cluster_name', 'user_count', 
                                       'avg_messages_per_day', 'avg_influence_score']]
    
    # Sort by influence (top users first)
    if 'influence_score' in final_report.columns:
        final_report = final_report.sort_values('influence_score', ascending=False)
    
    return final_report, cluster_summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <whatsapp_chat_file.txt> [n_clusters] [random_state]")
        sys.exit(1)
    
    chat_file = sys.argv[1]
    n_clusters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    random_state = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    result_df = analyze_chat(chat_file, n_clusters=n_clusters, random_state=random_state)
    print(f"\nAnalysis complete! Results for {len(result_df)} users.")
