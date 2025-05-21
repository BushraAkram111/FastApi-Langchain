import streamlit as st
import base64
from openai import OpenAI
import requests
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPPORTED_IMAGE_TYPES = [
    "image/png", "image/jpeg", "image/jpg", "image/webp", 
    "image/gif", "image/bmp", "image/svg+xml"
]

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    st.title("AI Image Processor")
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        app_mode = st.radio("Select Mode:", 
                           ["Image to Text", "Text to Image"],
                           index=0)
        
        if app_mode == "Image to Text":
            st.subheader("Image Input Settings")
            option = st.radio("Input Method:", 
                            ("Upload Image", "Image URL"),
                            horizontal=True)
            
            if option == "Upload Image":
                uploaded_image = st.file_uploader("Choose an image", 
                                                 type=["png", "jpg", "jpeg", "webp"])
            else:
                image_url = st.text_input("Image URL:")
            
            # Advanced options
            with st.expander("Advanced Settings"):
                custom_prompt = st.text_area("Custom Prompt:", 
                                           "Carefully observe the image and give a complete, intelligent description. Include objects, background, setting, lighting, people's clothing/expressions, and anything meaningful you can infer.")
                model_version = st.selectbox("Model Version:", ["gpt-4o", "gpt-4-vision-preview"])
        
        elif app_mode == "Text to Image":
            st.subheader("Image Generation Settings")
            # Advanced options
            with st.expander("Advanced Settings"):
                image_size = st.selectbox("Image Size:", 
                                        ["1024x1024", "1024x1792", "1792x1024"])
                image_quality = st.selectbox("Quality:", ["standard", "hd"])
                image_style = st.selectbox("Style:", ["vivid", "natural"])
    
    # Main content area
    if app_mode == "Image to Text":
        st.header("Image to Text")
        
        if option == "Upload Image" and uploaded_image:
            if uploaded_image.type not in SUPPORTED_IMAGE_TYPES:
                st.error(f"Unsupported image format: {uploaded_image.type}")
            else:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                image_data = f"data:{uploaded_image.type};base64,{encode_image(uploaded_image)}"
        elif option == "Image URL" and image_url:
            try:
                st.image(image_url, caption="Image from URL", use_column_width=True)
                image_data = image_url
            except:
                st.error("Couldn't load image from this URL")
                image_data = None
        else:
            image_data = None
        
        if st.button("Generate Description", type="primary"):
            if not image_data:
                st.error("Please upload an image or provide an image URL.")
            else:
                try:
                    with st.spinner("🔍 Analyzing image..."):
                        ai_response = client.chat.completions.create(
                            model=model_version,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": custom_prompt},
                                        {"type": "image_url", "image_url": {"url": image_data}},
                                    ],
                                }
                            ]
                        )
                        description = ai_response.choices[0].message.content.strip()
                    
                    with st.expander("📝 Image Description", expanded=True):
                        st.write(description)
                
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
    
    elif app_mode == "Text to Image":
        st.header("Text to Image")
        
        prompt = st.text_area("Describe the image you want to generate:", 
                            placeholder="A beautiful sunset over mountains with a lake reflection")
        
        if st.button("Generate Image", type="primary"):
            if not prompt:
                st.error("Please enter a description for the image.")
            else:
                try:
                    with st.spinner("Creating your image..."):
                        response = client.images.generate(
                            model="dall-e-3",
                            prompt=prompt,
                            n=1,
                            size=image_size,
                            quality=image_quality,
                            style=image_style
                        )
                        image_url = response.data[0].url
                        
                        image_response = requests.get(image_url)
                        if image_response.status_code != 200:
                            st.error("Failed to download image from OpenAI.")
                        else:
                            st.image(image_url, caption="Generated Image", use_column_width=True)
                            
                            # Add download button
                            img_bytes = image_response.content
                            st.download_button(
                                label="⬇️ Download Image",
                                data=img_bytes,
                                file_name="generated_image.png",
                                mime="image/png",
                                use_container_width=True
                            )
                
                except Exception as e:
                    st.error(f"Failed to generate image: {str(e)}")

if __name__ == "__main__":
    main()
    # streamlit run image_to_text_to_image.py