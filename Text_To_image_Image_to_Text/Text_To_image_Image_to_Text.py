from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import base64
from openai import OpenAI
from fastapi.responses import JSONResponse
import requests
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from io import BytesIO
import requests  

client = OpenAI()
app = FastAPI()
client = OpenAI()
app = FastAPI()

SUPPORTED_IMAGE_TYPES = [
    "image/png", "image/jpeg", "image/jpg", "image/webp", 
    "image/gif", "image/bmp", "image/svg+xml"
]

@app.post("/Image-To-Text/")
async def generate_image_description(
    uploaded_image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    try:
        # Validate input
        if not uploaded_image and not image_url:
            return JSONResponse(
                content={"error": "Please upload an image or provide an image URL."},
                status_code=400
            )

        if uploaded_image:
            if uploaded_image.content_type not in SUPPORTED_IMAGE_TYPES:
                return JSONResponse(
                    content={"error": f"Unsupported image format: {uploaded_image.content_type}"},
                    status_code=400
                )
            image_data = f"data:{uploaded_image.content_type};base64,{encode_image(uploaded_image.file)}"
        else:
            image_data = image_url  # URL will be used directly

        prompt = (
            "Carefully observe the image and give a complete, intelligent description. "
            "Include objects, background, setting, lighting, people’s clothing/expressions, and anything meaningful you can infer."
        )

        request_payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}},
                    ],
                }
            ]
        }

        ai_response = client.chat.completions.create(**request_payload)
        description = ai_response.choices[0].message.content.strip()
        return {"description": description}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Something went wrong: {str(e)}"}
        )

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

@app.post("/Text-To-Image/")
async def generate_image_download(prompt: str = Form(...)):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url

        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            return JSONResponse(status_code=500, content={"error": "Failed to download image from OpenAI."})

        image_stream = BytesIO(image_response.content)

        return StreamingResponse(
            image_stream,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate image: {str(e)}"}
        )


# Image Encoding Helper
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# uvicorn main:app --reload
