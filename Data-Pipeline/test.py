import subprocess
import os

video_folder_path = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/COMBINED_DATASET/VIDEOS"
video_file_names = os.listdir(video_folder_path)
print(len(video_file_names))

for i in video_file_names:
    vid_file = os.path.join(video_folder_path, i)
    keyframe_folder = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/COMBINED_DATASET/keyframes_images/" + i[:-4]
    os.makedirs(keyframe_folder, exist_ok=True)  # Create the directory here, with exist_ok=True
    keyframe_images = os.path.join(keyframe_folder, 'frames_%03d.jpg')  # Path for keyframe images
    keyframe_txt = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/COMBINED_DATASET/keyframe_txt_folder/" + i[:-4] + '.txt'
    keyframes_timestamps = "/home/easgrad/baluhars/1234567890/Video_Dataset/SUNDAY_TEST/COMBINED_DATASET/keyframe_timestamps_folder/" + i[:-4] + "_timestamps.txt"

    # Define the command line code you want to run
    command1 = 'ffmpeg -i ' + vid_file + ' -vf "select=\'gt(scene,0.1)\',showinfo,setpts=N/FRAME_RATE/TB" -q:v 2 -vsync vfr -f image2 ' + keyframe_images + ' 2> ' + keyframe_txt
    command2 = "grep showinfo " + keyframe_txt + " | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > " + keyframes_timestamps

    # Run the command and capture its output
    subprocess.run(command1, shell=True)
    subprocess.run(command2, shell=True)