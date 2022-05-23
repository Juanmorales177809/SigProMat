import speech_recognition as sr
import time


r= sr.Recognizer()

with sr.AudioFile("audio1.wav") as source:
    audio = r.listen(source)
    try:
        print("Reading audio file. Please, wait a moment...")
        text= r.recognize_google(audio, language='es-Es')
        time.sleep(1.5)
        print(text)
        print(type(text))
    except:
        print('I am sorry, I can not understand')