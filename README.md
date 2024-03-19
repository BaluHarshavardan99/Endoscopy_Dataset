# Endoscopy_Dataset
Endoscopy Text-Image Pairs Dataset

For YouTube download: Use PyTube version 12.0.0

For extracting keyframes: 

ffmpeg -i video_test.mp4 -vf "select='gt(scene,0.1)',showinfo,setpts=N/FRAME_RATE/TB" -q:v 2 -vsync vfr -f image2 /home/easgrad/baluhars/PIPELINE/VIDEOS/test_frames/frames_%03d.jpg 2> keyframes_output.txt
