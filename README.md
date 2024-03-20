# Endoscopy Dataset: Text-Image Pair Dataset
Endoscopy Text-Image Pairs Dataset

# Files Description
1. YouTube_Audio_Video_Download.py -
2. clip_classification.py - 
3. youtube_urls.txt -
4. whisper_v3_audio2text.py - 

# Data Processing Steps
1. Selecting YouTube video
   * YouTube Data API (Not used, future work)
2. Downloading YouTube video
   * PyTube Library
3. Extracting Audio from video - moviepy
4. Extracting key fames from video (Hyper-parameter: 0 -1)
    * ffmpeg
    * **Extracting keyframes:**
       *ffmpeg -i video_test.mp4 -vf "select='gt(scene,0.1)',showinfo,setpts=N/FRAME_RATE/TB" -q:v 2 -vsync vfr -f image2 /home/easgrad/baluhars/PIPELINE/VIDEOS/test_frames/frames_%03d.jpg 2> keyframes_output.txt
    * **Timestamps:** grep showinfo keyframes_output.txt | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > keyframes_timestamps.txt

6. Classifying the key frames using CLIP - CLIP Model
7. Applying Quilt code on key frames to extract chunks (Hyper-parameter: pair_chunk_time)
    * Problem: “pair_chunk_time” (Currently working)
    * Raised the issue in GitHub and Mailed the authors – No Reply
8. Extracting relevant audio bits and applying ASR
    * Whisper v3 Large model
9. Text Correction
    * ChatGPT 4.0, Spacy
10. Classification of Text & Extraction of Imp. points in text
    * Few Shot Learning: Gathered Endoscopy Terms, relevant examples for corresponding terms
    * GPT 4.0 (test), Gemma (Implementation) - Currently working
11. Combining Text & Image
    * Building the CSV files - Currently working




For YouTube download: Use PyTube version 12.0.0


