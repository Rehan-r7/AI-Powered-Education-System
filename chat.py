import google.generativeai as genai
from fastapi import FastAPI, HTTPException


# from pydantic import BaseModel


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


MAX_CONTEXT_TOKENS = 5000  # Adjust based on your model

SYSTEM_PROMPT = "You are a helpful and friendly chatbot. Answer questions concisely and accurately. If you don't know the answer, say 'I don't know'."

conversation_history = []


# class ChatRequest(BaseModel):
#     message: str


app = FastAPI()


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


def get_gemini_response(user_input, conversation_history, system_prompt=None):
    """Gets a response from the Gemini API, managing conversation history and token limits."""

    # Construct the prompt
    prompt = ""
    if system_prompt:
        prompt += system_prompt + "\n\n"

    for turn in conversation_history:
        prompt += "User: " + turn["user"] + "\n"
        prompt += "Assistant: " + turn["model"] + "\n"

    prompt += "User: " + user_input + "\n"
    prompt += "Assistant:"  

    try:
        response = model.generate_content(prompt)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"Error during API call: {e}")
        raise HTTPException(status_code=500, detail="Error from Gemini API")


# @app.post("/chat")
# async def chat_endpoint(request: ChatRequest):
#     """Endpoint to handle user messages and return Gemini's response."""
#     global conversation_history  

#     user_message = request.message

#     conversation_history = truncate_history(conversation_history, MAX_CONTEXT_TOKENS, SYSTEM_PROMPT)

#     try:
#         model_response = get_gemini_response(user_message, conversation_history, SYSTEM_PROMPT)
#     except HTTPException as e:
#         raise e  # Re-raise the exception

#     conversation_history.append({"user": user_message, "model": model_response})

#     return {"response": model_response}


# @app.get("/clear_history")
# async def clear_history():
#     """Clears the conversation history (for testing purposes)."""
#     global conversation_history
#     conversation_history = []
#     return {"message": "Conversation history cleared."}
