
import os
import torch
import whisper
import pinecone
import numpy as np
import pandas as pd
from pytube import YouTube

def video_to_audio(video_url, destination):

	# Get the video
	video = YouTube(video_url)

	# Convert video to Audio
	audio = video.streams.filter(only_audio=True).first()

	# Save to destination
	output = audio.download(output_path = destination)

	name, ext = os.path.splitext(output)
	new_file = name + '.mp3'

	# Replace spaces with "_"
	new_file = new_file.replace(" ", "_")

	# Change the name of the file
	os.rename(output, new_file)

	print("File successfully downloaded: ", new_file)
  
	return new_file

audio_path = "audio_data"

list_videos = [ ]

# Create dataframe
transcription_df = pd.DataFrame(list_videos, columns=['URLs'])
transcription_df.head()

transcription_df["file_name"] = transcription_df["URLs"].apply(lambda url: video_to_audio(url, audio_path))
transcription_df.head()


# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
	# Load the model
	whisper_model = whisper.load_model("small", device=device)

	def audio_to_text(audio_file):
		return whisper_model.transcribe(audio_file, fp16=False)["text"]

	# Apply the function to all the audio files
	transcription_df["transcriptions"] = transcription_df["file_name"].apply(lambda f_name: audio_to_text(f_name))

	# Show the first five rows
	transcription_df.head()


	import textwrap
	wrapper = textwrap.TextWrapper(width=60)
	first_transcription = transcription_df.iloc[0]["transcriptions"]
	formatted_transcription = wrapper.fill(text=first_transcription)
	# Check first transcription
	print(formatted_transcription)
except Exception as e:
	print(e)