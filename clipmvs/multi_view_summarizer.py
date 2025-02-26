import cv2
import csv
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from clipmvs.clip_retriever import CLIPEmbeddingRetriever 
from clipmvs.qdrant_handler import QdrantHandler 
from clipmvs.video_loader import VideoDataLoader

class MultiViewSummarizer:
    """
    A class to handle multi-view summarization of videos using CLIP and Qdrant.
    """
    def __init__(self, config_path='config.json'):
        """
        Initialize the MultiViewSummarizer.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        self.retriever = CLIPEmbeddingRetriever()
        self.qdrant_handler = QdrantHandler(config_path)
        self.vidloaderdict = {}
        self.timestampvsframes = {}
    
    def set_video_loaders(self, video_paths, batch_size=100, interval=12):
        for vid_path in video_paths:
            self.vidloaderdict[vid_path] = VideoDataLoader(vid_path, batch_size, interval=interval)
            print(self.vidloaderdict[vid_path].get_total_frames(),self.vidloaderdict[vid_path].get_fps())
    
    def process_videos(self, video_paths, batch_size=100, interval=12):
        """
        Process multiple videos concurrently to extract and store CLIP embeddings.

        Args:
            video_paths (list): List of paths to the video files.
            batch_size (int): Number of frames to fetch in each batch.
            interval (int): Interval between frames to fetch.
        """
            
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_single_video, video_path, batch_size, interval) for video_path in video_paths]
            for future in futures:
                future.result()  # Wait for all futures to complete

    def process_single_video(self, video_path, batch_size, interval):
        """
        Process a single video to extract and store CLIP embeddings.

        Args:
            video_path (str): Path to the video file.
            batch_size (int): Number of frames to fetch in each batch.
            interval (int): Interval between frames to fetch.
        """
        video_loader = self.vidloaderdict[video_path]
        for frames, timestamps in video_loader:
            video_loader.visualize_images(frames)
            image_embeddings = self.retriever.get_CLIP_vision_embedding(frames)
            # (timestamps)
            metadata = [{"timestamp": ts, "path":video_path} for ts in timestamps]
            print(self.qdrant_handler.store_embedding(image_embeddings, metadata))

    def generate_summary(self, query, top_k=50, is_image=False):
        """
        Generate a multi-view summary for a given query.

        Args:
            query (str or PIL.Image.Image): The query text or image.
            top_k (int): The number of top results to retrieve.
            is_image (bool): Flag to indicate if the query is an image.

        Returns:
            list: A summary combining textual and visual information.
        """
        if is_image:
            query_embedding = self.retriever.get_CLIP_vision_embedding([query])[0]
        else:
            query_embedding = self.retriever.get_CLIP_text_embedding([query])[0]
        # print(query_embedding)
        results = self.qdrant_handler.query_embedding(query_embedding)
        # print(results)
        # Extract visual information
        summary = []
        for result in results:
            timestamp = result.payload['timestamp']
            frame =  self.vidloaderdict[result.payload['path']].get_frame_by_timestamp(timestamp)
            print(frame)
            if(timestamp not in self.timestampvsframes):
                self.timestampvsframes[timestamp] = [frame]
            else:
                self.timestampvsframes[timestamp] += [frame]
            summary.append({
                "timestamp": timestamp,
                "frame": frame,
                "similarity": result.score,
                "path":result.payload['path']
            })
        
        return summary

    def generate_and_visualize_summaries(self, queries, top_k=50, is_images=None):
        """
        Generate and visualize summaries for multiple queries.

        Args:
            queries (list): List of queries (texts or images).
            top_k (int): The number of top results to retrieve for each query.
            is_images (list): List of boolean values indicating if each query is an image.
        """
        if is_images is None:
            is_images = [False] * len(queries)
        
        all_summaries = []
        for i, query in enumerate(queries):
            summary = self.generate_summary(query, top_k=top_k, is_image=is_images[i])
            all_summaries+=summary
            self.visualize_summary(summary, query, is_images[i])
            csv_path = f'summary_{i}.csv'  # Create a CSV for each query
            self.save_summary_to_csv(summary, query, csv_path)
        
        self.export_vid(all_summaries)

    def visualize_summary(self, summary, query, is_image):
        """
        Visualize the summary generated.

        Args:
            summary (list): The summary data containing timestamps and frames.
            query (str or PIL.Image.Image): The query text or image.
            is_image (bool): Flag to indicate if the query is an image.
        """
        # Visualize frames
        plt.figure(figsize=(15, 5))
        print(summary)
        for i, item in enumerate(summary):
            frame = item["frame"]
            print(frame)
            timestamp = item["timestamp"]
            if frame:
                print(frame)
                plt.subplot(1, len(summary), i + 1)
                plt.imshow(frame)
                plt.title(f"Timestamp: {timestamp:.2f}s\nScore: {item['similarity']:.4f}")
                plt.axis('off')
        plt.suptitle(f'Summary for query: {"Image" if is_image else query}')
        plt.show()

        # Visualize timeline
        timestamps = [item["timestamp"] for item in summary]
        scores = [item["similarity"] for item in summary]

        plt.figure(figsize=(10, 2))
        plt.hlines(1, 0, max(timestamps), colors='gray', linestyles='dashed')
        plt.eventplot(timestamps, orientation='horizontal', colors='blue')
        for ts, score in zip(timestamps, scores):
            plt.text(ts, 1.05, f"{ts:.2f}s\n{score:.4f}", ha='center', va='bottom', fontsize=8)
        plt.title(f'Queried Vectors on Timeline for query: {"Image" if is_image else query}')
        plt.xlabel('Timestamp (s)')
        plt.yticks([])
        plt.show()

    def save_summary_to_csv(self, summary,query, csv_path):
        """
        Save the summary metadata to a CSV file.

        Args:
            summary (list): The summary data containing timestamps and frames.
            csv_path (str): Path to the output CSV file.
        """
        fieldnames = ['query','timestamp', 'similarity','video_path']
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for item in summary:
                writer.writerow({'query':query,'timestamp': item['timestamp'], 'similarity': item['similarity'], 'video_path': item['path']})
    
    def export_vid(self,all_summaries, fps=12):
        images={}
        print(all_summaries)
        for item in all_summaries: 
            images[item['timestamp']] = item['frame']
        image_index = list(images.keys())
        image_index.sort()
        width, height = list(images.values())[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("./output.mp4", fourcc, fps, (width, height))

        for x in image_index:
            np_image = np.array(images[x])
            bgr_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            out.write(bgr_image)
        out.release()

    def close(self):
        """
        Close the Qdrant connection.
        """
        self.qdrant_handler.close()
