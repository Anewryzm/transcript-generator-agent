import io
import base64
import modal
from pydantic import BaseModel
from typing import Optional

image = modal.Image.debian_slim(python_version="3.11").pip_install(
  "chatterbox-tts==0.1.1",
  "fastapi[standard]",
  "transformers",
  "accelerate",
  "pydantic",
  )

def load_model(and_return: bool = False):
  from chatterbox.tts import ChatterboxTTS

  model = ChatterboxTTS.from_pretrained(device="cuda")

  if and_return:
    return model

app = modal.App("chatterbox-tts-api", image=image)

with image.imports():
  import torchaudio as ta
  from chatterbox.tts import ChatterboxTTS

# Create a volume for caching the model
cache_dir = "/cache"
# Create a Modal volume to store the model cache
model_cache = modal.Volume.from_name("chatterbox-tts-cache", create_if_missing=True)
# Mount the volume to the image
image = image.run_function(load_model, volumes={cache_dir: model_cache})

@app.cls(gpu="a10g", volumes={cache_dir: model_cache}, scaledown_window=60*5, enable_memory_snapshot=True)
class ChatterboxTSS:
  @modal.enter()
  def init(self):
    self.model = load_model(and_return=True)
  
  @modal.method()
  def generate_audio(self, prompt: str, audio_prompt_path: str = None):
    wav = self.model.generate(prompt, audio_prompt_path=audio_prompt_path)
    buffer = io.BytesIO()

    ta.save(buffer, wav, self.model.sr, format="wav")
    buffer.seek(0)
    
    # Convert to base64 string (easily serializable)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

class ChatterboxTTSAPI(BaseModel):
  prompt: str
  audio_prompt_path: Optional[str] = None

# Web API for Chatterbox TTS
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
  "fastapi[standard]",
  "pydantic",
  "python-multipart"
)

# FastAPI endpoint as an ASGI app
@app.function(
  image=web_image,
  # limit the number of concurrent requests to avoid overloading the model
  max_containers=1
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def web_api():
  from fastapi import FastAPI, Response
  import base64

  api = FastAPI(title="Chatterbox TTS API")
  model = ChatterboxTSS()

  @api.post("/generate")
  async def generate_audio(request: ChatterboxTTSAPI):
    prompt = request.prompt
    audio_prompt_path = request.audio_prompt_path

    try:
      # Call the remote method synchronously to avoid async generator issues
      if not audio_prompt_path:
        base64_audio = model.generate_audio.remote(prompt=prompt)
      else:
        base64_audio = model.generate_audio.remote(prompt=prompt, audio_prompt_path=audio_prompt_path)
      
      # Decode the base64 string back to bytes
      audio_bytes = base64.b64decode(base64_audio)
      
      # Return as a regular Response
      return Response(
        content=audio_bytes,
        media_type="audio/wav"
      )
    except Exception as e:
      return Response(
        content=f"Error generating audio: {str(e)}",
        status_code=500
      )
  
  return api