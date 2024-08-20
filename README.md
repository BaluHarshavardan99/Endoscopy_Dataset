# Endoscopy Dataset: Text-Image Pair Dataset
Endoscopy Text-Image Pairs Dataset

The code for the Dataset Extraction is present in Data-Pipeline folder.


# Data Processing Steps
1. Filtering Endoscopy videos
2. Downloading YouTube video and extracting audio
   * PyTube Library
   * For YouTube download: Use PyTube version 12.0.0
   * Extracting Audio from video - moviepy
3. Extracting key fames from video (Hyper-parameter range: 0-1, chosen value: 0.01)
    * FFmpeg
    * **Extracting keyframes:** ffmpeg -i video_test.mp4 -vf "select='gt(scene,0.01)',showinfo,setpts=N/FRAME_RATE/TB" -q:v 2 -vsync vfr -f image2 /home/easgrad/baluhars/PIPELINE/VIDEOS/test_frames/frames_%03d.jpg 2> keyframes_output.txt
    * **Timestamps:** grep showinfo keyframes_output.txt | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > keyframes_timestamps.txt

4. Classifying the key frames using CLIP and Endoscopy Classifier
    * CLIP Model
    * Endoscopy Classifier
5. Applying chunking algorithm on key frames to extract chunks (Hyper-parameter: pair_chunk_time, chosen value: 10)
6. Extracting relevant audio chuks and applying ASR
    * Whisper-v3-large model
7. Text Correction
    * Spacy, ChatGPT 4.0 API
8. Combining Text & Image
    * Building the CSV files







