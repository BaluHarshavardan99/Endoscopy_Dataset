import subprocess
import torch
import clip
from PIL import Image
import os
from pydub import AudioSegment
from faster_whisper import WhisperModel


import csv

def clip_classification(images_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # Path to the folder containing images
    folder = images_path
    # Get the list of image files in the folder
    image_files = [os.path.join(folder, file) for file in os.listdir(folder)]
    predictions = []
    image_paths = []  # List to store image paths
    for image_path in image_files:
        image_paths.append(image_path)  # Append image path
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(["a presentation slide on endoscopy techniques", "an endoscopic image from endoscopy camera"]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            #print("probs is ......... {}" .format(probs))
        
        # Find the highest probability in the list of probs
        max_prob = max(probs[0])

        # Find the index of the highest probability
        index = list(probs[0]).index(max_prob)
        #print("index is ........................................ {}" .format(index))
        predictions.append(index)
    #print("predictions is ......................... {}" .format(predictions))
    return predictions, image_paths  # Return image paths along with predictions
    
def extract_segment(audio_path, start_time, end_time, output_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)
    
    # Extract the segment based on start and end times
    segment = audio[start_time * 1000:end_time * 1000]  # Convert times to milliseconds
    
    # Export the segment to a new file
    segment.export(output_path, format="mp3")

# def clip_audio2text(audio):
#     model_size = "large-v3"
#     # Initialize the WhisperModel
#     model = WhisperModel(model_size, device="cpu", compute_type="int8")

#     # Transcribe audio and get segments
#     segments, info = model.transcribe(
#         audio,
#         beam_size=5,
#         vad_filter=True,
#         vad_parameters=dict(min_silence_duration_ms=500),
#     )

#     # Open a file to save the text
#     with open("/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATA-102/audio_bits/output_text.txt", "w") as f:        
#         # Write each segment to the file
#         for segment in segments:
#             #f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
#             f.write("%s\n" % (segment.text))

#     #print("Transcribed text saved to 'T_saginata.txt'")



import os

def clip_audio2text(audio):
    model_size = "large-v3"
    # Initialize the WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    # Transcribe audio and get segments
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Define the output file path
    output_file_path_txt = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/audio_bits/output_text.txt"

    # Create the file if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path_txt), exist_ok=True)

    # Open the file in write mode
    with open(output_file_path_txt, "w") as f:
        # Write each segment to the file
        for segment in segments:
            f.write("%s\n" % (segment.text))

    print("Transcribed text saved to 'output_text.txt'")




def get_histo_srt_im_recon(times_rows, preds, pair_chunk_time):
    running_t = 0
    srt_se_times = []
    srt_s = 0
    srt_e = 0
    chunk_im = []
    chunk_im_time = []
    temp_im = []
    temp_im_time = []
    temp_se = []
    prev_time = 0
    prev_idx = 0
    prev_prd = 0

    for idx, (row, prd) in enumerate(zip(times_rows, preds)):
        if (not row):  # or (not prd):
            # i.e  == 0
            prev_time = float(row)  # Convert to float
            prev_idx = idx
            prev_prd = prd
            continue

        if prd and (not prev_prd):
            srt_s = max([float(prev_time), float(row) - pair_chunk_time])  # Convert to float

            temp_im.append(idx)
            temp_im_time.append(row)
        elif prd and prev_prd:
            if float(row) - float(prev_time) > pair_chunk_time:  # Convert to float
                # end run
                chunk_im.append(temp_im or [idx])
                chunk_im_time.append(temp_im_time or [row])

                temp_im = []
                temp_im_time = []
                temp_im.append(idx)
                temp_im_time.append(row)
                running_t = 0

                srt_e = row
                temp_se.extend([srt_s, srt_e])
                srt_se_times.append(temp_se)
                temp_se = []
                srt_s = float(row) - pair_chunk_time  # Convert to float
            else:
                running_t += float(row) - float(prev_time)  # Convert to float
                if running_t > pair_chunk_time:
                    temp_im.append(idx)
                    temp_im_time.append(row)

                    chunk_im.append(temp_im)
                    chunk_im_time.append(temp_im_time)

                    temp_im = []
                    temp_im_time = []
                    running_t = 0

                    srt_e = row
                    temp_se.extend([srt_s, srt_e])
                    srt_se_times.append(temp_se)
                    temp_se = []
                    srt_s = float(row) - pair_chunk_time  # Convert to float
                else:
                    # inbetween
                    temp_im.append(idx)
                    temp_im_time.append(row)
        elif (not prd) and prev_prd:
            srt_e = row
            temp_se.extend([srt_s, srt_e])
            srt_se_times.append(temp_se)
            temp_se = []

            if not temp_im:
                temp_im.append(prev_idx)
                temp_im_time.append(prev_time)
            chunk_im.append(temp_im)
            chunk_im_time.append(temp_im_time)

            temp_im = []
            temp_im_time = []
            running_t = 0
        elif (not prd) and (not prev_prd):
            pass

        prev_time = float(row)  # Convert to float
        prev_idx = idx
        prev_prd = prd
    return srt_se_times, chunk_im, chunk_im_time


video_folder_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/VIDEOS"
video_file_names = os.listdir(video_folder_path)
print(len(video_file_names))

audio_folder_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/AUDIO"
audio_file_names = os.listdir(audio_folder_path)
#print(video_file_names)

# Define the CSV file path
csv_file_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/results.csv"
text_csv_file_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/text_csv_file.csv"
image_paths_csv_file_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/image_paths.csv"


# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    # Create a CSV writer object
    writer = csv.writer(file)
    # Write the header row
    # writer.writerow(['S No', 'vid_id', 'keyframes_times', 'predictions', 'chunks', 'chunks_images', 'chunks_image_times'])
    writer.writerow(['S No', 'vid_id', 'keyframes_times', 'predictions', 'chunks', 'chunks_images', 'chunks_image_times', 'image_paths'])
    with open(image_paths_csv_file_path, mode='w', newline='') as image_paths_file:
        # Create a CSV writer object
        image_paths_writer = csv.writer(image_paths_file)
        # Write the header row
        image_paths_writer.writerow(['vid_id', 'chunk_id', 'image_path'])
        with open(text_csv_file_path, mode='w', newline='') as text_file:
            # Create a CSV writer object
            text_writer = csv.writer(text_file)
            
            # Write the header row
            text_writer.writerow(['vid_id', 'chunk_id', 'text'])
        
            # Iterate over each video file
            COUNTER = 0
            for i, video_name in enumerate(video_file_names, start=1):
                if COUNTER < 103:
                    print("Processing and extracting key frames from video ................. {} - video number: {}" .format(video_name, i))
                    timestamps_txt = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/keyframe_timestamps_folder/"+ video_name[:-4] +"_timestamps.txt"
                    
                    # Initialize an empty list to hold the numbers
                    timestamps = []

                    # Open the file and read the numbers
                    with open(timestamps_txt, 'r') as f:
                        for line in f:
                            # Strip whitespace and newlines, then add the number to the list
                            timestamps.append(line.strip())

                    vid_id = video_name[:-4]
                    keyframes = timestamps
                    clip_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/keyframes_images/"
                    predictions, image_paths = clip_classification(clip_path + video_name[:-4])
                    #print("Predictions: ", predictions)

                    chunks, chunks_images, chunks_image_times = get_histo_srt_im_recon(keyframes, predictions, 10)
                    # print("Chunk_images is ...................: ", chunks_images)

                    # Write the data to the CSV file
                    writer.writerow([i, vid_id, keyframes, predictions, chunks, chunks_images, chunks_image_times, image_paths])

                    # Write the image paths associated with chunks to the CSV file
                    for chunk_id, chunk in enumerate(chunks_images):
                        # print("chunk_id: ", chunk_id)
                        # print("chunk: ", chunk)
                        for a in chunk:
                            chunk_image_paths = image_paths[a] 
                            image_paths_writer.writerow([vid_id, chunk_id, chunk_image_paths])

                    print("CSV file for video: {} created successfully!" .format(video_name))          

                      
                    # Get the chunks for the video
                    for chunk_id, chunk in enumerate(chunks):
                        print("Extracting text ............ for video: {} and chunk: {}" .format(video_name, chunk_id))
                        # Example usage
                        audio_path = audio_folder_path + "/" + video_name[:-4] + ".mp3"
                        
                        start_time = float(chunk[0])  # Start time in seconds
                        end_time = float(chunk[1])    # End time in seconds
                        #print("start and end times are ............................... {} and {}" .format(type(start_time), type(end_time)))
                        
                        # Extract audio segment
                        output_audio_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/audio_bits/output_segment.mp3"
                        extract_segment(audio_path, start_time, end_time, output_audio_path) 
                        
                        # Transcribe audio segment
                        clip_audio2text(output_audio_path) 
                        
                        # Read the transcribed text from the text file
                        with open('/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/NEW-DATASET-2/audio_bits/output_text.txt', 'r') as text_file:
                            text_out = text_file.read().strip()  # Read and strip whitespace
                        
                        # Write the data to the output CSV file
                        text_writer.writerow([vid_id, chunk_id, text_out])
                    
                    print("Text CSV file created successfully! for video: {}" .format(video_name))
                    COUNTER += 1
                else:
                    break

print("All videos processed successfully!")



        
    




 