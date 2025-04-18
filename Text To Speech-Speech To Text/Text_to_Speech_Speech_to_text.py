# Import necessary modules from FastAPI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse  # For returning JSON responses
from fastapi.middleware.cors import CORSMiddleware  # To allow frontend and backend to communicate
from pydantic import BaseModel  # For validating input data
import shutil  # For file operations like saving
import os  # For creating directories
from openai import OpenAI  # OpenAI client for Whisper and TTS APIs

# Initialize the FastAPI app
app = FastAPI()

# Initialize OpenAI client (it will use your API key from the environment)
client = OpenAI()

# Create a directory to temporarily store uploaded audio files
UPLOAD_DIR = "uploaded_audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # If folder exists, do not raise an error

# Define the allowed types of audio files
ACCEPTABLE_AUDIO_TYPES = [
    "audio/mpeg",  # mp3
    "audio/mp4",   # mp4
    "audio/x-m4a", # m4a
    "audio/mpg",   # mpga
    "audio/wav",   # wav
    "audio/webm"   # webm
]

# Enable CORS (Cross-Origin Resource Sharing)
# This allows your frontend (e.g., React app) to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domain in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input format for the Text-to-Speech endpoint
class TTSRequest(BaseModel):
    text: str  # Text to be converted to speech
    voice: str = "fable"  # Voice style (e.g., alloy, echo, fable, etc.)
    model: str = "tts-1"  # TTS model name (you can also use "tts-1-hd")

# Text-to-Speech endpoint: converts text to audio
@app.post("/text-to-speech/")
async def text_to_speech(input: TTSRequest):
    """
    This endpoint receives text from the user and generates a spoken audio file using OpenAI's Text-to-Speech model.
    """

    try:
        # Generate audio from text using OpenAI's TTS
        response = client.audio.speech.create(
            model=input.model,
            voice=input.voice,
            input=input.text
        )

        # Save the audio file locally
        file_path = "output.mp3"
        response.stream_to_file(file_path)

        # Return a message and file path
        return {"message": "TTS processing completed", "file_path": file_path}

    except Exception as e:
        # Handle and return errors
        raise HTTPException(status_code=500, detail=f"Error processing TTS: {str(e)}")

# Speech-to-Text endpoint: converts audio files to text
@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile = File(...)):
    """
    This endpoint takes an audio file from the user and converts the spoken words into written text using OpenAI's Whisper model.
    """

    # Check if the uploaded file type is acceptable
    if file.content_type not in ACCEPTABLE_AUDIO_TYPES:
        return JSONResponse(
            content={"error": f"Invalid file type: {file.content_type}. Only audio files are allowed."},
            status_code=400
        )

    try:
        # Save the uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use OpenAI's Whisper to transcribe the audio
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.translations.create(
                model="whisper-1",  # Using Whisper model
                file=audio_file,
                response_format="text"
            )

        # Delete the temporary file after processing
        os.remove(file_path)

        # Return the transcription result as JSON
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        # If something goes wrong, return a 500 error with the error message
        raise HTTPException(status_code=500, detail=f"Error processing transcription: {str(e)}")


# This block only runs if you run the file directly (e.g., python main.py)
if __name__ == "__main__":
    import uvicorn
    # Start the FastAPI server on localhost at port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
