#DEPENDENCIES
import speech_recognition as sr
import sentiment as s

#RECORD SOUND
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    #RECOGNISE AUDIO
    text = r.recognize_google(audio)
    print(text)
try:
    #PRINT THE SENTIMENT AS WELL AS CONFIDENCE
    print(s.sentiment(text))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))