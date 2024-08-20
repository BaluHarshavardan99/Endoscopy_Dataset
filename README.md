# Endoscopy Text-Image Pairs Dataset

This repository contains code for extracting and processing endoscopy text-image pairs, focusing on the filtering, extraction, and classification of data from endoscopy videos.

## Data Pipeline Overview

The dataset extraction code is located in the `Data-Pipeline` folder.

### Data Processing Steps

1. **Filtering Endoscopy Videos**
    - Preprocess and filter videos to focus on relevant endoscopy content.

2. **Downloading YouTube Videos and Extracting Audio**
    - **Library:** PyTube
    - **PyTube Version:** 12.0.0 (required for compatibility)
    - **Audio Extraction:** Use `moviepy` to extract audio from the downloaded videos.

3. **Extracting Keyframes from Video**
    - **Library:** FFmpeg
    - **Hyperparameter (scene threshold):** 0.01
    - **Command:**
      ```bash
      ffmpeg -i video_test.mp4 -vf "select='gt(scene,0.01)',showinfo,setpts=N/FRAME_RATE/TB" -q:v 2 -vsync vfr -f image2 /home/easgrad/baluhars/PIPELINE/VIDEOS/test_frames/frames_%03d.jpg 2> keyframes_output.txt
      ```
    - **Extracting Timestamps:**
      ```bash
      grep showinfo keyframes_output.txt | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > keyframes_timestamps.txt
      ```

4. **Classifying Keyframes**
    - **Models:** CLIP, Endoscopy Classifier
    - Classify extracted keyframes using the models mentioned above.

5. **Chunking Keyframes**
    - **Algorithm:** Chunking algorithm
    - **Hyperparameter (`pair_chunk_time`):** 10 (seconds)
    - Apply chunking to keyframes to create meaningful image-text pairs.

6. **Extracting Relevant Audio Chunks and Applying ASR**
    - **Model:** Whisper-v3-large
    - Extract audio chunks corresponding to keyframes and apply Automatic Speech Recognition (ASR).

7. **Text Correction**
    - **Libraries:** SpaCy, ChatGPT 4.0 API
    - Correct the extracted text using NLP tools and ChatGPT.

8. **Combining Text & Image**
    - Create CSV files by combining the processed text and corresponding images.

## Usage

1. Clone this repository.
2. Navigate to the `Data-Pipeline` folder.
3. Follow the steps outlined in the data processing section to generate the dataset.

## Requirements

- PyTube 12.0.0
- moviepy
- FFmpeg
- CLIP Model
- Whisper-v3-large
- SpaCy
- ChatGPT 4.0 API

Ensure all dependencies are installed before running the code.

## License

This project is licensed under the MIT License.

## Acknowledgements

Special thanks to the contributors and the open-source community for the tools and libraries used in this project.
