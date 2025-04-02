import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse


# Imports from helper.py
from helper import transcribe_video, get_quiz_question


app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class TranscriptStorage:
    def __init__(self):
        self.transcript: Optional[str] = None

    def set_transcript(self, new_transcript: str):
        self.transcript = new_transcript

    def get_transcript(self) -> Optional[str]:
        return self.transcript



transcript_storage = TranscriptStorage() 


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video, generates transcript, and stores it globally."""
    try:
        file_ext = file.filename.split(".")[-1].lower()
        allowed_extensions = {"mp4", "avi", "mov", "mkv", "flv"}

        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type")

        file_path = UPLOAD_DIR /  f"uploaded_video.{file_ext}"

        with file_path.open("wb") as buffer:
            buffer.write(await file.read())

        # Generate and store transcript
        transcript = transcribe_video(str(file_path))
        transcript_storage.set_transcript(transcript)  # Store in the global instance

        return {
            "filename": file.filename,
            "file_path": str(file_path),
            "transcript": transcript
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# ---------------------------------------------Chat Fucntionality---------------------------------------------


from pydantic import BaseModel

from helper import truncate_history, get_gemini_response


class ChatRequest(BaseModel):
    message: str


MAX_CONTEXT_TOKENS = 8192  # Adjust based on model pricing and performance


conversation_history = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handles chat requests and fetches relevant transcript dynamically."""
    global conversation_history

    # Retrieve the latest transcript
    transcript = transcript_storage.get_transcript()  # Use the global instance
   
    if not transcript:
        return {"response": "I currently do not have any transcript. Please upload a video first."}
   
    SYSTEM_PROMPT = f"""You are an AI assistant specialized in answering questions about a video's content based on a timestamped transcript: {transcript}. The transcript consists of dialogue with precise time markers, allowing you to provide accurate, **detailed, and well-explained** responses.

    **Your Responsibilities:**  

    1. **Extract and Explain Information from the Transcript in Detail:**  
    - Use the provided transcript to extract relevant details based on the user's question.  
    - **Expand on key points** using context from the video rather than providing brief answers.  
    - If the answer spans multiple timestamps, **summarize comprehensively**, ensuring all important details are included.  

    2. **Provide In-Depth and Well-Structured Answers:**  
    - If the transcript provides relevant information, **explain it fully** rather than giving a short response.  
    - **Where applicable, include direct excerpts** from the transcript to support your answer.  
    - Provide **additional context or explanations** to enhance the user's understanding.  

    3. **Reference Timestamps and Quotes for Clarity:**  
    - Mention timestamps when they help the user locate specific details.  
    - If the user asks for exact wording, provide a direct **quoted excerpt** from the transcript.  
    - When possible, summarize related portions of the transcript to provide a broader perspective.  

    4. **Handle Questions Beyond the Transcript Thoughtfully:**  
    - If the transcript does not contain the requested information, **first try to answer using prior knowledge** while keeping the response relevant to the video’s topic.  
    - If prior knowledge is insufficient, politely inform the user while suggesting related information when possible.  

    
    5. **Provide Output in Properly Formatted Markdown:**  
    - Structure responses using **headings, bullet points, bold text, and code blocks**.  
    - Use **blockquotes (`>`) for direct quotes** from the transcript.  
    - Format responses for readability to enhance user experience. 

    **Example Interactions (Encouraging Detailed Responses):**  

    ✅ **User:** What does the speaker say about AI assistants?  
    ✔️ **Assistant:** The speaker discusses AI assistants in detail at multiple points in the video. At [0:00:07 - 0:00:15], they explain that AI assistants help with scheduling tasks, reminders, and general queries. However, they contrast this with AI agents, which proactively take action based on learned patterns. For example, the speaker says, *"Assistants respond to requests, but agents anticipate needs before you're even aware of them."* This distinction suggests that AI assistants are more reactive, while AI agents operate autonomously.  

    ✅ **User:** What is the main topic of the video?  
    ✔️ **Assistant:** The video primarily focuses on the evolving roles of AI assistants and agents. The speaker explains how assistants, like Siri or Alexa, function by responding to user commands, while more advanced AI models act independently. For example, at [0:00:30 - 0:00:45], they mention how AI-powered systems are now being used in business automation, handling customer interactions without human intervention. This discussion aligns with broader AI trends and how automation is shaping industries.  

    ✅ **User:** Is it easy to use an LLM?  
    ✔️ **Assistant:** While the transcript does not specifically address this, LLMs are designed for ease of use, especially in conversational AI applications. Many platforms, such as ChatGPT, provide user-friendly interfaces that require no technical expertise. However, **ease of use depends on the application**—for instance, integrating an LLM into a software system may require programming skills, while using one in a chatbot is straightforward. If you're looking for a simple way to interact with an LLM, tools with natural language interfaces, as discussed in similar AI contexts within the video, can help bridge the gap.  

    🚫 **User:** What is the speaker’s name?  
    ⚠️ **Assistant:** The transcript does not mention the speaker’s name, so I cannot determine that information.  

    🚫 **User:** What happens after the video ends?  
    ⚠️ **Assistant:** My responses are based on the transcript provided, so I can't determine events beyond the video's content. However, if you’re asking about potential follow-ups, I can provide general insights based on the video's topic.  


    ### **Final Enhancements:**  
    ✅ **Markdown Formatting for Readability** – Ensures clear, structured answers with headings, lists, blockquotes, and code blocks.  
    ✅ **Longer, More Informative Responses** – Expands on key points rather than providing one-line answers.  
    ✅ **Stronger Use of Transcript Data** – Encourages quoting and summarizing relevant sections.  

    """


    user_message = request.message

    # Ensure conversation history doesn't exceed max tokens
    conversation_history = truncate_history(conversation_history, MAX_CONTEXT_TOKENS, SYSTEM_PROMPT)

    try:
        model_response = get_gemini_response(user_message, conversation_history, SYSTEM_PROMPT)
    except HTTPException as e:
        raise e  

    conversation_history.append({"user": user_message, "model": model_response})

    response = json.loads(model_response.strip())

    return JSONResponse(content=response)
    # return {"result": model_response}


@app.get("/clear_history")
async def clear_history():
    """Clears both conversation history and transcript."""
    global conversation_history
    conversation_history = []
    transcript_storage.set_transcript(None)  # Clear transcript
    return {"message": "Conversation history and transcript cleared."}


@app.get("/quiz")
async def generate_quiz():
    """Generates a quiz based on the latest transcript."""
    
    try : 

        transcript = transcript_storage.get_transcript()

        if not transcript:
            return {"response": "I currently do not have any transcript. Please upload a video first."}
        
        quiz_questions = get_quiz_question(transcript)
        # print(quiz_questions)

        return JSONResponse(content=quiz_questions)
        # return quiz_questions
    
    except Exception as e : 

        raise HTTPException(status_code=500, detail=str(e))

