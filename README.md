# WhatsApp Group Behavior Insight System

A machine learning system for analyzing group chat dynamics and generating explainable behavioral profiles from WhatsApp chat exports.

## Project Overview

Understanding communication patterns in group chats can provide insights into social dynamics, engagement levels, and behavioral roles. This system processes exported WhatsApp group chat data to identify distinct behavioral clusters and generate natural-language profiles for each participant.

The system addresses the challenge of extracting meaningful insights from large volumes of chat data by applying unsupervised learning techniques to identify behavioral patterns, then translating these patterns into human-readable descriptions.

## Key Features

- Automated parsing of WhatsApp chat export files (supports multiple date formats)
- Comprehensive feature engineering capturing temporal, linguistic, and engagement patterns
- KMeans clustering to identify distinct behavioral groups
- Natural language behavior profile generation for each user
- Influence score calculation based on activity, engagement, and conversation initiation
- CSV and PDF report generation with detailed analytics
- Interactive Streamlit web interface for non-technical users
- Command-line interface for batch processing and integration



## Project Structure

```
.
├── parser.py          # WhatsApp chat parsing (raw chat → DataFrame)
├── features.py        # Feature engineering
├── clustering.py      # Feature scaling + KMeans clustering
├── profiling.py       # Natural language behavior profiles
├── report.py          # CSV/PDF report generation
├── main.py            # Main entry point with analyze_chat() function
├── app.py             # Streamlit web interface
├── requirements.txt   # Python dependencies
└── README.md          # This file
```



## How It Works

The system follows a five-stage pipeline:

1. **Parsing**: Converts raw WhatsApp chat export text into structured data with timestamps, users, and messages
2. **Feature Engineering**: Extracts behavioral metrics including message frequency, response times, emotional expressiveness, link sharing, and conversation initiation patterns
3. **Clustering**: Applies KMeans clustering on standardized features to group users with similar behaviors
4. **Profiling**: Generates natural language descriptions of each user's behavior based on their cluster assignment and feature values
5. **Reporting**: Produces structured CSV reports and formatted PDF documents with analysis results

## Feature Engineering

The system extracts behavioral features across multiple dimensions:

**Temporal Patterns**: Average messages per day, response time between consecutive messages, and night activity ratio (messages between 10 PM and 6 AM)

**Message Characteristics**: Average and median message length, total characters sent

**Emotional Expressiveness**: Frequency of emojis, exclamation marks, question marks, and uppercase usage

**Engagement Metrics**: Link and resource sharing frequency, conversation initiation rate (identifying users who start new conversation threads after gaps)

**Activity Level**: Total message count and daily activity patterns

These features are aggregated per user and standardized before clustering to ensure all dimensions contribute equally to the analysis.

## Clustering Approach

The system uses KMeans clustering with a default of 5 clusters, though this is configurable. KMeans was selected for its interpretability, computational efficiency, and effectiveness with the behavioral feature space.

**Why KMeans**: The behavioral features form a continuous, multi-dimensional space where KMeans effectively identifies distinct behavioral archetypes. The algorithm's simplicity ensures results are reproducible and explainable.

**Why Fixed K**: A fixed number of clusters provides consistent, comparable results across different group chats. The default of 5 clusters balances granularity with interpretability, capturing major behavioral patterns without over-segmentation. Users can adjust this parameter in advanced settings for research purposes.

Features are standardized using StandardScaler to ensure all dimensions contribute equally to distance calculations, preventing high-magnitude features from dominating the clustering.

## Installation

### Requirements

- Python 3.8 or higher
- Required packages listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd behavioural-ml

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computations
- scikit-learn: Machine learning algorithms (KMeans, StandardScaler)
- reportlab: PDF report generation
- matplotlib: Visualization (for Streamlit app)
- wordcloud: Word cloud generation (for Streamlit app)
- streamlit: Web interface framework

## Usage

### Command Line Interface

The simplest way to analyze a chat file:

```bash
python main.py chat_export.txt
```

With custom parameters:

```bash
python main.py chat_export.txt 6 42
```

Where the arguments are: `[chat_file] [n_clusters] [random_state]`

The analysis generates two output files:
- `user_behavior_report.csv`: Structured data with all user metrics
- `whatsapp_user_behavior_report.pdf`: Formatted report with behavior profiles

### Python API

```python
from main import analyze_chat

# Basic usage with defaults
result_df = analyze_chat("chat_export.txt")

# Custom configuration
result_df = analyze_chat(
    "chat_export.txt",
    n_clusters=6,
    random_state=42,
    csv_output="custom_report.csv",
    pdf_output="custom_report.pdf"
)

# Access results
print(result_df[['user', 'cluster', 'influence_score', 'behavior_profile']])
```

### Streamlit Web Interface

For interactive analysis with a user-friendly interface:

```bash
streamlit run app.py
```

The web interface provides:
- File upload functionality
- Interactive user behavior profile viewer
- Cluster summary statistics
- Download buttons for generated reports
- Behavior word cloud visualization

Advanced settings (cluster count and random state) are available in a collapsible section for researchers who need to customize parameters.

## Example Insights

The system identifies several common behavioral archetypes:

**Regular Participants**: Moderate activity levels, balanced engagement, typical response times. These users contribute consistently without dominating conversations.

**Active Conversationalists**: High message frequency, quick responses, emotionally expressive. These users drive ongoing discussions and maintain group engagement.

**Information Broadcasters**: Frequent link sharing, high activity, often initiate conversations. These users serve as information hubs, sharing resources and starting topic discussions.

**Silent Observers**: Low message frequency, infrequent responses, minimal engagement. These users primarily consume content rather than actively participating.

**Night Owls**: Disproportionate activity during late night hours (10 PM - 6 AM), indicating different time zone or lifestyle patterns.

Each user receives an influence score (0-1) based on their activity level, response speed, and conversation initiation frequency, providing a quantitative measure of their role in group dynamics.

## Limitations

- **Date Format Support**: While the parser supports multiple date formats (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD), some WhatsApp export formats may not be recognized. Users should verify their export format matches supported patterns.

- **Small Groups**: The clustering algorithm requires sufficient data points. Groups with fewer than the number of clusters may produce unstable results.

- **Language Assumptions**: Feature extraction assumes English-language patterns (emoji detection works universally, but linguistic features are optimized for English).

- **Temporal Patterns**: Night activity detection uses fixed hours (10 PM - 6 AM) which may not account for all time zones or cultural patterns.

- **Cluster Interpretation**: The default cluster names (Regular Participants, Active Conversationalists, etc.) are optimized for 5 clusters. Using different cluster counts will produce generic cluster IDs.

- **Feature Stability**: Some features (like response time) may be less meaningful in very active groups where messages arrive rapidly.

## Ethics & Privacy Note

This system is designed for analyzing group chat data where all participants have consented to analysis, or for research purposes with appropriate ethical approval.

**Important Considerations**:

- Only analyze chats where you have permission from all participants
- Respect privacy: do not share individual behavioral profiles without consent
- Use insights responsibly: behavioral patterns should inform understanding, not judgment
- Consider context: behavioral profiles reflect chat activity patterns, not personality traits
- Data handling: chat exports contain personal information; handle files securely and delete after analysis if required

The system processes data locally and does not transmit information to external services. All analysis occurs on the user's machine.

## Future Improvements

- **Adaptive Clustering**: Implement methods to automatically determine optimal cluster count (e.g., elbow method, silhouette analysis)

- **Multi-language Support**: Enhance feature extraction to support non-English languages and cultural communication patterns

- **Temporal Analysis**: Add time-series analysis to track behavioral changes over time

- **Network Analysis**: Incorporate relationship mapping between users based on reply patterns and mentions

- **Export Format Detection**: Automatic detection and handling of additional WhatsApp export formats

- **Real-time Processing**: Support for streaming analysis of ongoing chats

- **Custom Profile Templates**: Allow users to define custom behavior profile templates for domain-specific insights

- **Statistical Validation**: Add significance testing and confidence intervals for behavioral metrics

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]
