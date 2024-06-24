# Downloader

This Downloader module is a Python package designed for downloading and transcribe video from YouTube. It provides command-line interfaces for easy usage and includes functionalities for handling requirements.

## Installation

To install this module, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Help

To use this module

``` bash
python3 -m downloader --help

usage: [-h] [--url URL] [--type TYPE] [--download DOWNLOAD] [--output_path OUTPUT_PATH] [--file_json FILE_JSON] [--transcribe TRANSCRIBE] [--model_dimension MODEL_DIMENSION]
```

## Usage

``` bash
  -h, --help        HELP                show this help message and exit
  --url             URL                 url of the video or playlist
  --type            TYPE                type of the url [playlist/video]
  --download        DOWNLOAD            download video and audio
  --download_video  DOWNLOAD_VIDEO      download only video
  --download_audio  DOWNLOAD_AUDIO      download only audio
  --output_path     OUTPUT_PATH         folder output
  --file_json       FILE_JSON           path to the json file
  --transcribe      TRANSCRIBE          transcribe the audio inside the video
  --model_dimension MODEL_DIMENSION     whisper model type
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
