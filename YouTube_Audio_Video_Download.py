# This file downloads YouTube videos and their audio files from a list of YouTube URLs in a .txt file

from pytube import YouTube

class YB_Downloader:
    def __init__(self):
        # super(CLASS_NAME, self).__init__(*args, **kwargs)
        
    def Download(self, link, name, tot, rem_files):
        yt = YouTube(
            link,
            use_oauth=True,
            allow_oauth_cache=True
        )

        # Get the highest resolution video stream
        video_stream = yt.streams.get_highest_resolution()

        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()
        print("Downloading video and audio for ......................{} [ {} / {} ] " .format(name, rem_files, tot))

        # Download the video
        video_stream.download(filename=self.data_dir + "/" + "VIDEO/" + name + ".mp4")

        # Download the audio
        audio_stream.download(filename=self.data_dir + "/" + "audio/" + name + ".mp3")

    def download_from_file(self, file_path):
        counter = 1
        with open(file_path, 'r') as file:
            total_videos = sum(1 for _ in file)
        with open(file_path, 'r') as file:
            for line in file:
                youtube_link = "https://www.youtube.com/watch?v=" + line.strip()
                remaining_files = total_videos - counter
                Download(youtube_link, line.strip(), total_videos, remaining_files)
                counter+=1

    #file_path = input("Enter the path to the .txt file containing YouTube video links: ")
    download_from_file(self.data_dir + "/" + "video_links.txt")
