import yt_dlp
import pathlib

def download_from_YT(video_url: str,
                     output_dir: pathlib.Path):
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best available audio
        'outtmpl': f"{output_dir}/%(title)s.%(ext)s",  # Output filename template
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # Convert to WAV
                'preferredquality': '192',  # Audio quality (192 kbps)
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        original_filepath = pathlib.Path(ydl.prepare_filename(info_dict)) 
        wav_filepath = original_filepath.with_suffix('.wav')  
        
        ydl.download([video_url])  
        
    return wav_filepath
