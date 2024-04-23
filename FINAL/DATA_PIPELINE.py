import subprocess
import torch
import clip
from PIL import Image
import os
from pydub import AudioSegment
from faster_whisper import WhisperModel


import csv


class AutoDataPipeline:
    def __init__(self, ):
        # super(CLASS_NAME, self).__init__()
        self.images_path

    def clip_classification(self, images_path):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        folder = images_path
        image_files = [os.path.join(folder, file) for file in os.listdir(folder)]
        predictions = []
        for image_path in image_files:
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
            predictions.append(index)
        return predictions
        

    def extract_segment(self, audio_path, start_time, end_time, output_path):
        audio = AudioSegment.from_file(audio_path)
        
        segment = audio[start_time * 1000:end_time * 1000]  # Convert times to milliseconds
        
        # Export the segment to a new file
        segment.export(output_path, format="mp3")


    def clip_audio2text(self, audio, model_size = "large-v3", output_path = self.data_dir + self.dataset_name + "/audio_bits/output_text.txt"):
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        segments, info = model.transcribe(
            audio,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        with open(output_path, "w") as f:
            # Write detected language and probability to the file
            f.write("Detected language '%s' with probability %f\n\n" % (info.language, info.language_probability))
            
            for segment in segments:
                #f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
                f.write("%s\n" % (segment.text))

    def get_histo_srt_im_recon(self, times_rows, preds, pair_chunk_time):
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

    def process(self):
        video_folder_path = "/root/Video_Dataset/SUNDAY_TEST/VIDEOS"
        video_file_names = os.listdir(video_folder_path)

        audio_folder_path = "/root/Video_Dataset/SUNDAY_TEST/AUDIO"
        audio_file_names = os.listdir(audio_folder_path)
        #print(video_file_names)

        csv_file_path = "/root/Video_Dataset/SUNDAY_TEST/results.csv"

        with open(csv_file_path, mode='w', newline='') as file:
            writer1 = csv.writer(file)
            writer1.writerow(['S No', 'vid_id', 'keyframes_times', 'predictions', 'chunks', 'chunks_images', 'chunks_image_times'])
            
            for i, video_name in enumerate(video_file_names, start=1):
                print("Processing and extracting key frames ................. {} - video number: {}" .format(video_name, i))
                timestamps_txt = self.args.data_dir + "/" + self.dataset_name + "/" + keyframe_timestamps_folder/"+ video_name[:-4] +"_timestamps.txt"
                
                timestamps = []

                with open(timestamps_txt, 'r') as f:
                    for line in f:
                        timestamps.append(line.strip())

                vid_id = video_name[:-4]
                keyframes = timestamps
                # clip_path = "/root/Video_Dataset/SUNDAY_TEST/keyframes_images/"
                predictions = self.clip_classification(self.args.data_dir + self.args.dataset_name + "/" + video_name[:-4])
                #print("Predictions: ", predictions)

                chunks, chunks_images, chunks_image_times = self.get_histo_srt_im_recon(keyframes, predictions, 10)

                writer1.writerow([i, vid_id, keyframes, predictions, chunks, chunks_images, chunks_image_times])
                print("CSV file for video: {} created successfully!" .format(video_name))

                
                text_csv_file_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/text_csv_file.csv"

                with open(text_csv_file_path, mode='w', newline='') as text_file:
                    text_writer = csv.writer(text_file)
                    
                    text_writer.writerow(['vid_id', 'chunk_id', 'text'])
                    
                        
                    # Get the chunks for the video
                    for chunk_id, chunk in enumerate(chunks, start=1):
                        print("Extracting text ............ for video: {} and chunk: {}" .format(video_name, chunk_id))
                        audio_path = audio_folder_path + "/" + video_name[:-4] + ".mp3"
                        
                        start_time = float(chunk[0])
                        end_time = float(chunk[1])
                        
                        output_audio_path = "/root/Video_Dataset/SUNDAY_TEST/audio_bits/output_segment.mp3"
                        extract_segment(audio_path, start_time, end_time, output_audio_path) 
                        
                        clip_audio2text(output_audio_path) 
                        
                        with open('/root/Video_Dataset/SUNDAY_TEST/audio_bits/output_text.txt', 'r') as text_file:
                            text_out = text_file.read().strip() \
                        
                        text_writer.writerow([vid_id, chunk_id, text_out])
                    
                    print("Text CSV file created successfully! for video: {}" .format(video_name))





        
