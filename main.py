import googleapiclient.discovery
from dotenv import load_dotenv
import os
import pandas as pd
from transformers import pipeline
import streamlit as st
import plotly.express as px

# Video ID
def get_video_id(link):
    video_id = link.split("=")[1]
    return video_id

# Comments on video
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
        comment_response = comment_request.execute()

        all_comments.extend(comment_response['items'])

        page_token = comment_response.get('nextPageToken')
        if not page_token:
            break

    return all_comments

# Video Stats
def get_video_stats(video_id, youtube):
    video_request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    video_response = video_request.execute()

    yt_page_name = video_response['items'][0]['snippet']['channelTitle']
    video_title = video_response['items'][0]['snippet']['title']
    view_count = video_response['items'][0]['statistics']['viewCount']
    like_count = video_response['items'][0]['statistics']['likeCount']
    comments_count =  video_response['items'][0]['statistics']['commentCount']

    return yt_page_name, video_title, view_count, like_count, comments_count

# Comments Dataframe
def get_dataframe(comments):
    comments_list = []

    for comment in comments:
        top_level_comment = comment['snippet']['topLevelComment']
        author_name = top_level_comment['snippet']['authorDisplayName']
        comment_text = top_level_comment['snippet']['textDisplay']
        like_count = top_level_comment['snippet']['likeCount']

        comments_dict = {}
        comments_dict['author'] = author_name
        comments_dict['text'] = comment_text
        comments_dict['likes'] = like_count
        comments_list.append(comments_dict)
        df_comments = pd.DataFrame(comments_list)
    
    return df_comments

# Sentiment Analysis
def get_sentiments(df):
    model = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    results = {}
    for index, row in df.iterrows():
        try:
            row_text = row['text']
            row_author = row['author']
            sentiment = model(row_text)[0]['label']
            results[row_author] = sentiment
        except RuntimeError:
            continue
    # Merging the sentiments and comments
    df_results = pd.DataFrame([results]).T.reset_index().rename(columns={'index':'author', 0:'sentiment'})
    df_results = df_results.merge(df, how='left')
    return df_results

# CSS for centering headers
st.markdown(
    """
    <style>
    .centered {
        text-align: center;
    }
    .centered {
        text-align: center;
        padding: 0 0 2rem 0;
    }
    .info-text {
        text-align: center;
        color: #fff;
        margin-bottom: 0.5rem;
    }
    .info-header {
        text-align: center;
        color: #fff;
        margin-bottom: 2rem;
    }
    .video-container {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    load_dotenv('.gitignore/.env')
    api_key = os.getenv("YT_API")
    # Authenticate with the API
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    st.title("YouTube Video Sentiment Analyzer")
    st.write("Analyze comments from a YouTube video to get sentiment insights. Simply provide the link below! (UN Sustainable Development Goals - Overview Video Provided as Example Below)")

    # Input section
    link = st.text_input(
        "YouTube Video Link", 
        placeholder="Enter the video link here", 
        help="Paste a valid YouTube video URL.", 
        value="https://www.youtube.com/watch?v=M-iJM02m_Hg"
    )

    if st.button("Generate Analysis"):
        with st.spinner("Analyzing..."):

            # Extract video information and perform analysis
            try:
                video_id = get_video_id(link)
                all_comments = get_all_comments(video_id=video_id, youtube=youtube)
                df_comments = get_dataframe(all_comments)
                df_sentiment = get_sentiments(df_comments)
                df_results = df_sentiment['sentiment'].value_counts().to_frame().reset_index()
                df_results.columns = ['Sentiment', 'Count']

                yt_page_name, video_title, view_count, like_count, comments_count = get_video_stats(
                    video_id=video_id, youtube=youtube
                )

                # Create container for video details
                st.markdown("<div class='video-container'>", unsafe_allow_html=True)

                # Display video details
                st.markdown("<h2 class='centered'>Video Information</h2>", unsafe_allow_html=True)
                
                col1, spacing, col2 = st.columns([1, 0.2, 1])

                with col1:
                    st.markdown("<p class='info-text'>Channel Name</p>", unsafe_allow_html=True)
                    st.markdown(f"<h3 class='info-header'>{yt_page_name}</h3>", unsafe_allow_html=True)

                with col2:
                    st.markdown("<p class='info-text'>Video Title</p>", unsafe_allow_html=True)
                    st.markdown(f"<h3 class='info-header'>{video_title}</h3>", unsafe_allow_html=True)

                
                # Create three columns for metrics with spacing
                metric_col1, space1, metric_col2, space2, metric_col3 = st.columns([1, 0.1, 1, 0.1, 1])

                with metric_col1:
                    st.markdown("<p class='info-text'>Views</p>", unsafe_allow_html=True)
                    st.markdown(f"<h3 class='info-header'>{view_count}</h3>", unsafe_allow_html=True)

                with metric_col2:
                    st.markdown("<p class='info-text'>Likes</p>", unsafe_allow_html=True)
                    st.markdown(f"<h3 class='info-header'>{like_count}</h3>", unsafe_allow_html=True)
                    
                
                with metric_col3:
                    st.markdown("<p class='info-text'>Comments</p>", unsafe_allow_html=True)
                    st.markdown(f"<h3 class='info-header'>{comments_count}</h3>", unsafe_allow_html=True)
                    


                st.markdown("<div class='video-container'>", unsafe_allow_html=True)

                # Display sentiment analysis
                st.markdown("<h2 class='centered'>Sentiment Analysis Results</h2>", unsafe_allow_html=True)
                
                fig = px.bar(
                    df_results,
                    x='Sentiment',
                    y='Count',
                    color='Sentiment',
                    color_discrete_map={
                        'positive': '#4CAF50',
                        'negative': '#F44336',
                        'neutral': '#FFC107'
                    },
                    text='Count'
                )
                fig.update_layout(
                    title={
                        'text':'Comment Sentiment Distribution',
                        'font':{'size':28, 'family':'bold'},
                        'x': 0.5,
                        'xanchor': 'center',
                        'y': 0.95
                    },
                    xaxis=dict(
                        title='Sentiment',
                        title_font={'size': 22, 'family':'bold'},
                        tickfont={'size': 18, 'family':'bold'}
                    ),
                    yaxis=dict(
                        title='Number of Comments',
                        title_font={'size': 22, 'family':'bold'},
                        tickfont={'size': 18, 'family':'bold'}
                    ),

                    showlegend=False,
                    plot_bgcolor='white',
                    height=600,
                    width=900
                )

                fig.update_traces(
                    width=0.6,
                    textposition='outside',  # Position the text above the bars
                    textfont=dict(size=22, color='black', family='bold')   # Make the text larger
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Enter a valid YouTube Video Link")
        



if __name__ == "__main__":
    main()