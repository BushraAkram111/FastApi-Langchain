import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
import base64

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Custom CSS for beautiful styling
st.markdown("""
<style>
    :root {
        --primary: #4f46e5;
        --secondary: #f9fafb;
        --accent: #10b981;
    }
    .main {
        background-color: var(--secondary);
    }
    .sidebar .sidebar-content {
        background-color: white;
        padding: 1rem;
        border-right: 1px solid #e5e7eb;
    }
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-1px);
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .result-box {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid var(--accent);
    }
    .audio-player {
        width: 100%;
        margin: 1rem 0;
    }
    h1, h2, h3 {
        color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar with all options
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Voice+AI", width=150)
        st.title("Settings")
        
        # App mode selection
        app_mode = st.radio(
            "Select Mode:",
            ["🗣️ Speech to Text", "🔊 Text to Speech"],
            index=0
        )
        
        if app_mode == "🗣️ Speech to Text":
            st.subheader("Audio Input Settings")
            input_method = st.radio(
                "Input Method:",
                ["Upload Audio File", "Record Audio"],
                help="Choose how to provide audio input"
            )
            
            with st.expander("Advanced Settings"):
                language = st.selectbox(
                    "Language",
                    ["Auto Detect", "English", "Spanish", "French", "German", "Hindi"],
                    index=0
                )
                temperature = st.slider(
                    "Creativity", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.7,
                    help="Higher values make output more random"
                )
        
        elif app_mode == "🔊 Text to Speech":
            st.subheader("Voice Settings")
            voice = st.selectbox(
                "Voice Style",
                ["Alloy", "Echo", "Fable", "Onyx", "Nova", "Shimmer"],
                index=0
            )
            
            model = st.selectbox(
                "Model",
                ["tts-1 (Standard)", "tts-1-hd (High Quality)"],
                index=0
            )
            
            speed = st.slider(
                "Speech Speed",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
    
    # Main content area
    st.title("🎙️ AI Voice Studio")
    st.markdown("Transform between speech and text with cutting-edge AI")
    
    if app_mode == "🗣️ Speech to Text":
        st.subheader("Convert Audio to Text")
        
        if input_method == "Upload Audio File":
            audio_file = st.file_uploader(
                "Upload an audio file",
                type=["mp3", "wav", "m4a", "mp4", "webm"],
                help="Supported formats: MP3, WAV, M4A, MP4, WEBM"
            )
            
            if audio_file:
                st.audio(audio_file, format=audio_file.type.split('/')[1])
                
                if st.button("Transcribe Audio"):
                    with st.spinner("🔍 Analyzing audio content..."):
                        try:
                            # Save to temp file
                            temp_file = "temp_audio." + audio_file.name.split(".")[-1]
                            with open(temp_file, "wb") as f:
                                f.write(audio_file.getbuffer())
                            
                            # Transcribe audio
                            with open(temp_file, "rb") as audio_file_obj:
                                transcription = client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file_obj,
                                    response_format="text",
                                    temperature=temperature,
                                    language=None if language == "Auto Detect" else language.lower()
                                )
                            
                            # Clean up
                            os.remove(temp_file)
                            
                            # Display results
                            with st.container():
                                st.success("Transcription Complete!")
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                st.write(transcription)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
        
        else:  # Record Audio
            st.warning("Audio recording requires browser microphone access and is not fully supported in this demo.")
    
    else:  # Text to Speech
        st.subheader("Convert Text to Speech")
        
        text_input = st.text_area(
            "Enter your text:",
            placeholder="Type or paste the text you want to convert to speech...",
            height=200
        )
        
        if st.button("Generate Speech"):
            if not text_input:
                st.warning("Please enter some text first")
            else:
                with st.spinner("🎙️ Generating natural sounding speech..."):
                    try:
                        response = client.audio.speech.create(
                            model="tts-1" if "Standard" in model else "tts-1-hd",
                            voice=voice.lower(),
                            input=text_input,
                            speed=speed
                        )
                        
                        # Convert to bytes for Streamlit
                        speech_bytes = BytesIO(response.read())
                        
                        # Display audio player
                        st.audio(speech_bytes, format="audio/mp3")
                        
                        # Add download button
                        st.download_button(
                            label="⬇️ Download Audio",
                            data=speech_bytes,
                            file_name=f"tts_{voice.lower()}.mp3",
                            mime="audio/mp3"
                        )
                    
                    except Exception as e:
                        st.error(f"Error generating speech: {str(e)}")

if __name__ == "__main__":
    main()
    # streamlit run speech_app.py