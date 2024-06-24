import whisper
from downloader.logger import logger
import pandas as pd
import os


def transcribe(model_dimension, audio_path, output):
    model = whisper.load_model(model_dimension)
    transcript_dict = whisper.transcribe(model, audio_path, verbose=False)
    logger.info("Generate file transcript")
    file_name = (
        os.path.join(output, os.path.basename(audio_path).split(".")[0]) + ".txt"
    )
    with open(file_name, "w") as f:
        for segment in transcript_dict["segments"]:
            f.write(
                f"{round(segment['start'], 3)} -- {round(segment['end'], 3)} -- {segment['text']}\n"
            )

    return file_name


def convert_txt_to_xlsx(txt_path, xlsx_path):
    logger.info("Start conversion to xlsx")
    with open(txt_path, "r", encoding="utf-8") as txt_file:
        lines = txt_file.readlines()

    data = [line.strip().split("--") for line in lines]

    df = pd.DataFrame(data)
    df.to_excel(xlsx_path, index=False, header=False)
