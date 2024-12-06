# YouTube Comment Sentiment Analyzer

## Description
The YouTube Comment Sentiment Analyzer is an interactive web application that performs sentiment analysis on YouTube video comments. This tool helps content creators and analysts understand audience reactions by analyzing the emotional tone of comments and presenting the results through an intuitive dashboard. The application provides detailed video statistics and visualizes sentiment distribution using interactive charts.

## Key Features
- Real-time YouTube comment extraction
- Sentiment analysis using advanced NLP models
- Interactive data visualization
- Comprehensive video statistics
- User-friendly interface
- Responsive design

## Tech Stack
- **Frontend Framework**: Streamlit
- **Data Processing**: Pandas
- **API Integration**: Google YouTube Data API v3
- **Natural Language Processing**: Hugging Face Transformers
- **Data Visualization**: Plotly Express
- **Authentication**: Python-dotenv
- **Machine Learning Model**: CardiffNLP Twitter RoBERTa Model

## Code Structure and Implementation Details

### 1. Video Data Extraction
```python
def get_video_id(link):
    video_id = link.split("=")[1]
    return video_id

def get_video_stats(video_id, youtube):
    video_request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    # Returns video metadata including title, view count, likes, etc.
```
This section handles the extraction of video metadata using the YouTube Data API.

### 2. Comment Collection
```python
def get_all_comments(video_id, youtube):
    all_comments = []
    page_token = None

    while True:
        comment_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token
        )
        # Collects all comments with pagination support
```
Implements pagination to fetch all comments from the video.

### 3. Data Processing
```python
def get_dataframe(comments):
    comments_list = []
    for comment in comments:
        top_level_comment = comment['snippet']['topLevelComment']
        # Extracts relevant comment data and creates a DataFrame
```
Transforms raw comment data into a structured DataFrame for analysis.

### 4. Sentiment Analysis
```python
def get_sentiments(df):
    model = pipeline('sentiment-analysis', 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    # Analyzes sentiment of each comment using RoBERTa model
```
Utilizes the CardiffNLP Twitter RoBERTa model for accurate sentiment classification.

### 5. Visualization
```python
fig = px.bar(
    df_results,
    x='Sentiment',
    y='Count',
    color='Sentiment',
    color_discrete_map={
        'positive': '#4CAF50',
        'negative': '#F44336',
        'neutral': '#FFC107'
    }
)
```
Creates interactive visualizations using Plotly Express with custom styling.

