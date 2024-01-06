import numpy as np
import json as js
import os
import speech_recognition as rm
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/path/to/file.json"





class SpeecHandler():

    def __init__(self) -> None:
        
        self.max_seq_len = 1000
        self.texts = []

    def simple_audio_recognition(self):

        rec = rm.Recognizer()
        mic = rm.Microphone()

        with mic as sequence:
            
            audio = rec.listen(sequence)
            text = rec.recognize_vosk(audio, language="ru")
            self.texts.append(text)
        
        print(self.texts, f"[длина тензора слов: {len(self.texts)}]")
        

sp_handler_obj = SpeecHandler()
sp_handler_obj.simple_audio_recognition()





