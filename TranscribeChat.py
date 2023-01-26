import speech_recognition as sr
import openai
import os
from dotenv import load_dotenv

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set")
# obtain audio from the microphone
r = sr.Recognizer()
while True:
    
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
   
    # recognize speech using whisper
    
    try:
        
        text = r.recognize_whisper(audio, language="english")
        if "Jarvis" in text:
                    print("Question: " + text)
                    openai.api_key = os.getenv("OPENAI_API_KEY")

                
                    response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt="You are an AI assistant named Jarvis. The assistant is helpful, creative, clever, and very friendly. When i refer to jarvis i am refering to you. I am going to ask you questions, have conversations and talk to you. You should reply back to me appropriatly to the questions i ask. If the question is on a speciifc topic respond as an expert on the matter.Keep you answers to a minimum." + text,
                        temperature=0.7,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )

                    text = response['choices'][0]['text']
                    print("Response: " + text)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper")
    except KeyboardInterrupt:
            break



