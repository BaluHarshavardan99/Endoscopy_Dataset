import os
from moviepy.editor import VideoFileClip

def extract_audio_from_videos(input_folder, output_folder):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Add more video formats if needed
            video_path = os.path.join(input_folder, filename)
            audio_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.mp3")

            # Load the video file
            video_clip = VideoFileClip(video_path)

            # Extract the audio
            audio_clip = video_clip.audio

            # Write the audio to a file
            audio_clip.write_audiofile(audio_path)

            # Close the clips
            video_clip.close()
            audio_clip.close()

            print(f"Extracted audio from {filename} to {audio_path}")

# Define the input and output folders
input_folder = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/COMBINED_DATASET/VIDEOS"
output_folder = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/COMBINED_DATASET/AUDIO"

# Extract audio from videos
extract_audio_from_videos(input_folder, output_folder)
