import argparse
import pathlib
import mod.utils as utils
import mod.whisper as whis

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",
                        "-v",
                        type=str,
                        required=True)
    parser.add_argument("--output",
                        "-o",
                        type=pathlib.Path,
                        required=True)   
    
    return parser.parse_args()


def main(args):
    print(args)
    
    audio_file = utils.download_from_YT(args.video,
                           args.output)
    whisperModel = whis.Whisper("openai/whisper-medium")
    audio_features = whisperModel.loadData(audio_file)
    transcription = whisperModel.transcribe(audio_features)
    print(transcription)
    



args = parse_args()
main(args)