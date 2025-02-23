from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, AutoProcessor, SeamlessM4Tv2Model
from datasets import Dataset, Audio
import torch
import pathlib


class Whisper:
    def __init__(self,
                 model
                 ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model
        
    def createPipeline(self,
                       ) -> pipeline:
        pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                chunk_length_s=30,
                device=self.device,
                generate_kwargs={"language": "chinese"}
                )
        return pipe

    def loadData(self,
                audio_path: pathlib.Path
                 )-> dict:
        dataset = Dataset.from_dict({
            "audio": [str(audio_path)]
            }).cast_column("audio", Audio(sampling_rate=16_000))
        
        audio_sample = dataset[0]["audio"]

        # input_features = self.processor(audio_sample["array"],
        #                                 sampling_rate=audio_sample["sampling_rate"],
        #                                 return_tensors="pt"
        #                                 ).input_features
        return audio_sample
    
    def transcribe(self,
                   audio_features
                   ):
        # # generate token ids
        # predicted_ids = self.model.generate(audio_features,
        #                                     forced_decoder_ids=self.forced_decoder_ids)
        # # decode token ids to text
        # transcription = self.processor.batch_decode(predicted_ids,
        #                                             skip_special_tokens=True)
        
        pipe = self.createPipeline()
        transcription = pipe(audio_features.copy(), batch_size=8)
        
        return transcription



# class Seamlessm4TV2:
#     def __init__(self):
#         self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
#         self.model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
