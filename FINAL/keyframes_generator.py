import subprocess
import os



class FrameGenerator:
    def __init__(self, video_folder_path = self.args.data_dir + "/Video_Dataset/SUNDAY_TEST/VIDEOS/"):
        # super(CLASS_NAME, self).__init__(*args, **kwargs)
        self.video_folder_path = video_folder_path
        print(len(video_file_names))

    def process(self):
        video_file_names = os.listdir(self.video_folder_path)
        for i in video_file_names:
            vid_file = os.path.join(video_folder_path, i)
            keyframe_folder = self.args.data_dir + "/" + self.dataset_name + "/" + "keyframes_images/" + i[:-4]
            os.makedirs(keyframe_folder, exist_ok=True)  
            keyframe_images = os.path.join(keyframe_folder, 'frames_%03d.jpg')  # Path for keyframe images
            keyframe_txt = self.args.data_dir + "/" + self.dataset_name + "/" + "/keyframe_txt_folder/" + i[:-4] + '.txt'
            keyframes_timestamps = self.args.data_dir + "/" + self.dataset_name + "/" + keyframe_timestamps_folder/" + i[:-4] + "_timestamps.txt"

            command1 = 'ffmpeg -i ' + vid_file + ' -vf "select=\'gt(scene,0.1)\',showinfo,setpts=N/FRAME_RATE/TB" -q:v 2 -vsync vfr -f image2 ' + keyframe_images + ' 2> ' + keyframe_txt

            command2 = "grep showinfo " + keyframe_txt + " | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > " + keyframes_timestamps

            subprocess.run(command1, shell=True)
            subprocess.run(command2, shell=True)
