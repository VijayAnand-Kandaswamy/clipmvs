from PIL import Image
from clipmvs.multi_view_summarizer import MultiViewSummarizer

# Initialize the summarizer
summarizer = MultiViewSummarizer(config_path='../config.json') 

# Process videos to store embeddings
video_paths = ['vid1 path', 'vid2 path']  # Add more video paths as needed
summarizer.set_video_loaders(video_paths)
summarizer.process_videos(video_paths)

# Define multiple queries (both text and image)
queries = [
    "test query 1",
    "test query 2",
    Image.open("path to image query")
]
is_images = [False, False, True] 

# Generate and visualize summaries for multiple queries
summarizer.generate_and_visualize_summaries(queries,top_k=50, is_images=is_images)

# Close the Qdrant connection
summarizer.close()
