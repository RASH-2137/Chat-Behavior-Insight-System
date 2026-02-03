"""
Minimal Streamlit app for WhatsApp group behavior analysis
Wraps the analyze_chat() function from main.py
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import subprocess
from pathlib import Path
from main import analyze_chat
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Basic page setup
st.set_page_config(page_title="GroupChat Behavior Analysis", layout="wide")

st.title("Group chat Behavior Analysis")

# Let user upload their chat file
st.header("Upload Chat File")
uploaded_file = st.file_uploader(
    "Choose a WhatsApp chat export file (.txt)",
    type=['txt'],
    help="Export your WhatsApp group chat and upload the .txt file here"
)

# Demo mode: Try with sample data
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("**Or try the system with synthetic demo data:**")
with col2:
    try_sample = st.button("Try with Sample Data", type="secondary")

# Initialize session state for sample data analysis
if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = False

if try_sample:
    with st.spinner("Generating sample chat data..."):
        # Generate sample chat file
        try:
            from generate_sample_chat import generate_sample_chat
            sample_file = "sample_chat.txt"
            generate_sample_chat(sample_file)
            st.success("Sample data generated successfully! Running analysis...")
            # Set flag to use sample file
            st.session_state.use_sample_data = True
            # Clear any uploaded file
            uploaded_file = None
        except Exception as e:
            st.error(f"Failed to generate sample data: {str(e)}")
            st.session_state.use_sample_data = False

# Note about synthetic data
if try_sample or st.session_state.use_sample_data:
    st.info("ℹ️ Sample data is fully synthetic and provided only for demonstration. No real conversations are used.")

# Advanced Settings - hidden by default, uses defaults when collapsed
# Set default values first
n_clusters = 6
random_state = 42

# Advanced settings expander (collapsed by default)
with st.expander("Advanced Settings (For Researchers)", expanded=False):
    st.warning("Changing these parameters may alter cluster stability and behavior interpretation.")
    
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=20,
            value=6,
            help="Number of behavioral clusters to identify"
        )
    with col2:
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            help="For reproducible results"
        )

# Run analysis when button is clicked (either uploaded file or sample data)
should_analyze = False
file_to_analyze = None

# Handle uploaded file analysis
if uploaded_file is not None:
    # Clear sample data flag when user uploads a file
    st.session_state.use_sample_data = False
    if st.button("Analyze Chat", type="primary"):
        should_analyze = True
        # Save uploaded file to temp location (analyze_chat needs a file path)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_to_analyze = tmp_file.name

# Handle sample data analysis (auto-trigger when sample data is generated)
if st.session_state.use_sample_data and os.path.exists("sample_chat.txt"):
    should_analyze = True
    file_to_analyze = "sample_chat.txt"

# Execute analysis
if should_analyze and file_to_analyze:
        
        try:
            # Create temp folder for the output files
            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = os.path.join(tmp_dir, "user_behavior_report.csv")
                pdf_path = os.path.join(tmp_dir, "whatsapp_user_behavior_report.pdf")
                
                # Show spinner while processing
                with st.spinner("Analyzing chat... This may take a moment."):
                    # Call the main analysis function
                    result_df = analyze_chat(
                        file_path=file_to_analyze,
                        n_clusters=n_clusters,
                        random_state=random_state,
                        csv_output=csv_path,
                        pdf_output=pdf_path
                    )
                
                # Make sure we actually got results
                if result_df is None or result_df.empty:
                    st.error("Analysis completed but no results were generated.")
                else:
                    st.success("Analysis complete!")
                    # Reset sample data flag after successful analysis
                    if file_to_analyze == "sample_chat.txt":
                        st.session_state.use_sample_data = False
                    
                    # Show the results
                    st.header("Results Preview")
                    
                    # Quick stats at the top
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Users", len(result_df))
                    with col2:
                        st.metric("Clusters", result_df['cluster'].nunique())
                    with col3:
                        if 'influence_score' in result_df.columns:
                            avg_influence = result_df['influence_score'].mean()
                            st.metric("Avg Influence Score", f"{avg_influence:.2f}")
                    
                    # Show top users table
                    st.subheader("User Analysis (Top 10)")
                    preview_cols = ['user', 'cluster', 'total_messages', 'messages_per_day', 'influence_score']
                    available_cols = [col for col in preview_cols if col in result_df.columns]
                    st.dataframe(
                        result_df[available_cols].head(10),
                        use_container_width=True
                    )
                    
                    # Summary by cluster
                    if 'cluster' in result_df.columns:
                        st.subheader("Cluster Summary")
                        cluster_summary = result_df.groupby('cluster').agg({
                            'user': 'count',
                            'messages_per_day': 'mean' if 'messages_per_day' in result_df.columns else 'count',
                            'influence_score': 'mean' if 'influence_score' in result_df.columns else 'count'
                        }).reset_index()
                        cluster_summary.columns = ['Cluster', 'User Count', 'Avg Messages/Day', 'Avg Influence']
                        st.dataframe(cluster_summary, use_container_width=True)
                    
                    # NEW: User Behavior Profiles viewer
                    st.header("User Behavior Profiles")
                    
                    # Figure out which column has the profiles (could be either name)
                    profile_col = None
                    if 'cluster_aware_profile' in result_df.columns:
                        profile_col = 'cluster_aware_profile'
                    elif 'behavior_profile' in result_df.columns:
                        profile_col = 'behavior_profile'
                    
                    if profile_col and 'user' in result_df.columns:
                        # Get all users and sort them
                        user_list = result_df['user'].unique().tolist()
                        user_list.sort()
                        
                        # Dropdown to pick a user
                        selected_user = st.selectbox(
                            "Select a user to view their behavior profile",
                            user_list,
                            help="Choose a user from the list to see their detailed behavior analysis"
                        )
                        
                        if selected_user:
                            # Grab that user's row
                            user_data = result_df[result_df['user'] == selected_user].iloc[0]
                            
                            # Show cluster and influence in two columns
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                # Show cluster name if we have it
                                if 'cluster_name' in result_df.columns:
                                    cluster_name = user_data.get('cluster_name', 'N/A')
                                    st.info(f"**Cluster:** {cluster_name}")
                                elif 'cluster' in result_df.columns:
                                    cluster_id = user_data.get('cluster', 'N/A')
                                    st.info(f"**Cluster ID:** {cluster_id}")
                            
                            with info_col2:
                                # Show their influence score
                                if 'influence_score' in result_df.columns:
                                    influence = user_data.get('influence_score', 0)
                                    st.info(f"**Influence Score:** {influence:.2f}")
                            
                            # Show the full behavior profile text
                            profile_text = user_data.get(profile_col, 'No profile available.')
                            if profile_text and profile_text != 'No profile available.':
                                st.markdown("### Behavior Profile")
                                st.markdown(f"_{profile_text}_")
                            else:
                                st.warning("No behavior profile available for this user.")
                    else:
                        st.info("Behavior profiles are not available in the results.")
                    
                    # Download section
                    st.header("Download Reports")
                    
                    # CSV download button
                    if os.path.exists(csv_path):
                        with open(csv_path, 'rb') as f:
                            csv_data = f.read()
                        st.download_button(
                            label="Download CSV Report",
                            data=csv_data,
                            file_name="user_behavior_report.csv",
                            mime="text/csv"
                        )
                    
                    # PDF download button
                    if os.path.exists(pdf_path):
                        with open(pdf_path, 'rb') as f:
                            pdf_data = f.read()
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name="whatsapp_user_behavior_report.pdf",
                            mime="application/pdf"
                        )
                    
                    # Word cloud at the end - visualizes behavior patterns with emojis
                    st.header("Behavior Word Cloud")
                    
                    # Check which profile column we have
                    profile_col_for_wc = None
                    if 'cluster_aware_profile' in result_df.columns:
                        profile_col_for_wc = 'cluster_aware_profile'
                    elif 'behavior_profile' in result_df.columns:
                        profile_col_for_wc = 'behavior_profile'
                    
                    if profile_col_for_wc:
                        # Get all the profile texts
                        all_profiles = result_df[profile_col_for_wc].dropna().astype(str).tolist()
                        
                        if all_profiles:
                            # Regex to find emojis in the text
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
                            
                            # Collect all emojis and meaningful words
                            emoji_list = []
                            word_list = []
                            
                            for profile in all_profiles:
                                # Pull out any emojis
                                emojis = emoji_pattern.findall(profile)
                                emoji_list.extend(emojis)
                                
                                # Extract words (4+ chars, lowercase)
                                words = re.findall(r'\b[a-z]{4,}\b', profile.lower())
                                # Filter out common/meaningless words
                                stop_words = {'this', 'user', 'belongs', 'group', 'that', 'with', 'from', 'their', 'they', 'them', 'have', 'been', 'more', 'than', 'less', 'often', 'frequently', 'responds', 'writes', 'prefers', 'shares', 'active', 'reserved', 'expressive'}
                                meaningful_words = [w for w in words if w not in stop_words]
                                word_list.extend(meaningful_words)
                            
                            # Count how often each emoji/word appears
                            emoji_counts = Counter(emoji_list)
                            word_counts = Counter(word_list)
                            
                            # Build text for word cloud (repeat by frequency)
                            cloud_text_parts = []
                            # Top emojis, repeat them based on frequency
                            for emoji, count in emoji_counts.most_common(20):
                                cloud_text_parts.extend([emoji] * min(count, 10))  # Max 10 repeats per emoji
                            
                            # Top words, repeat them based on frequency
                            for word, count in word_counts.most_common(30):
                                cloud_text_parts.extend([word] * min(count, 5))  # Max 5 repeats per word
                            
                            cloud_text = ' '.join(cloud_text_parts)
                            
                            if cloud_text.strip():
                                # Generate and show the word cloud
                                try:
                                    wordcloud = WordCloud(
                                        width=800,
                                        height=400,
                                        background_color='white',
                                        colormap='viridis',
                                        max_words=100,
                                        relative_scaling=0.5,
                                        collocations=False
                                    ).generate(cloud_text)
                                    
                                    # Display it
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    ax.imshow(wordcloud, interpolation='bilinear')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    st.info("Word cloud generation skipped (insufficient data or display issue).")
                            else:
                                st.info("Not enough behavioral data to generate word cloud.")
                        else:
                            st.info("No profile data available for word cloud.")
                    else:
                        st.info("Behavior profiles not available for word cloud generation.")
        
        except ValueError as e:
            # Handle file format issues
            error_msg = str(e)
            if "No messages found" in error_msg or "file format" in error_msg.lower():
                st.error(f"Invalid file format: {error_msg}")
            elif "users" in error_msg.lower() or "too few" in error_msg.lower():
                st.error(f"Insufficient data: {error_msg}")
            else:
                st.error(f"Error: {error_msg}")
        
        except Exception as e:
            # Catch any other errors
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Please check that your file is a valid WhatsApp chat export.")
        
        finally:
            # Clean up the temp file we created (only if it was an uploaded file)
            if file_to_analyze and file_to_analyze != "sample_chat.txt" and os.path.exists(file_to_analyze):
                os.unlink(file_to_analyze)

else:
    st.info("👆 Please upload a WhatsApp chat export file (.txt) to begin analysis.")
