import pandas as pd
from downloader.logger import logger
import os
from moviepy.video.io.VideoFileClip import VideoFileClip


def cut_videos(transcript_folder, video_folder, output_folder=None):
    # Trova i file xlsx con lo stesso nome nella directory locale
    transcript_files = os.listdir(transcript_folder)
    video_files = os.listdir(video_folder)

    transcript_bases = {os.path.splitext(file)[0]: file for file in transcript_files}
    video_bases = {os.path.splitext(file)[0]: file for file in video_files}

    for base_name, xlsx_file in transcript_bases.items():
        if base_name in video_bases:
            video_file = video_bases[base_name]
            print(f"Matching video for {xlsx_file}: {video_file}")

            # Leggi il file XLSX per ottenere i tempi di start e end
            xlsx_path = os.path.join(transcript_folder, xlsx_file)
            df = pd.read_excel(xlsx_path, header=None)

            # Per ogni riga nella trascrizione
            for index, row in df.iterrows():
                start_time = row[0]  # Start
                end_time = row[1]  # End
                video_path = os.path.join(video_folder, video_file)

                # Carica il video e taglia il video
                video_clip = VideoFileClip(video_path)
                output_filename = f"{base_name}_{index}.mp4"  # Aggiungi un numero di sequenza al nome del file
                output_path = os.path.join(output_folder, output_filename)
                crop_video(video_clip, start_time, end_time, output_path)


def crop_video(video, start_time, end_time, output_path):
    subclip = video.subclip(start_time, end_time)
    subclip.write_videofile(output_path)
    subclip.close()
