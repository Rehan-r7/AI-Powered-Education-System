import torch
import warnings
from datetime import timedelta
import whisper_timestamped as whisper

warnings.simplefilter(action='ignore', category=FutureWarning)

def transcribe_video(video_path, model_size="base"):

    """
    Transcribes a video file and returns a formatted transcription with timestamps.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = whisper.load_model(model_size).to(device)
    audio = whisper.load_audio(video_path)
    result = whisper.transcribe(model, audio, language="en")

    def format_timestamp(seconds):

        """
        Convert seconds (float) to HH:MM:SS format using timedelta.
        """

        return str(timedelta(seconds=int(seconds))) 

  
    # transcription_text = "\n".join(
    #     f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n{segment['text'].strip()}\n"
    #     for segment in result['segments']
    # )
    
    transcription_text = " || ".join(
        f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}: {segment['text'].strip()}"
            for segment in result['segments']
    )

    return transcription_text  




# ------------------------------------------------------------------------------------- -------------------------------------------------------------------------------------

import google.generativeai as genai
from fastapi import HTTPException


genai.configure(api_key="AIzaSyAkzU0XWQgQRJlQezjXgcVLTfs2fx3exNQ")

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 1024,
  "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
)


def count_tokens(text):
    """Counts the number of tokens in a given text."""
    try:
        return model.count_tokens(text).total_tokens
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0 


def truncate_history(history, max_tokens, system_prompt=None):
    """Truncates the conversation history (FIFO) to stay within the token limit."""
    truncated_history = []
    current_tokens = 0

    if system_prompt:
        current_tokens += count_tokens(system_prompt)

    for turn in reversed(history):  
        turn_tokens = count_tokens(turn["user"]) + count_tokens(turn["model"])
        if current_tokens + turn_tokens <= max_tokens:
            truncated_history.insert(0, turn)  
            current_tokens += turn_tokens
        else:
            break  

    return truncated_history


import json

def get_gemini_response(user_input, conversation_history, system_prompt=None):
    """Gets a response from the Gemini API, managing conversation history and token limits."""

    # Construct the prompt
    prompt = ""
    if system_prompt:
        prompt += system_prompt + "<br><br>"

    for turn in conversation_history:
        prompt += "User: " + turn["user"] + "<br>"
        prompt += "Assistant: " + turn["model"] + "<br>"

    prompt += "User: " + user_input + "<br>"
    prompt += "Assistant:" 

    try:
        response = model.generate_content(prompt)
        response.resolve() 

       
        return response.text
        # try:
        #     json_response = json.loads(response.text)
        #     return json_response 
        # except json.JSONDecodeError:
          
        #     return {"response": response.text, "error": "Invalid JSON response"} 

    except Exception as e:
        print(f"Error during API call: {e}")
        raise HTTPException(status_code=500, detail=f"Error from Gemini API: {e}")
