import gradio as gr
import anthropic
import os
import asyncio
from dotenv import load_dotenv
from smolagents.mcp_client import MCPClient
from gradio import ChatMessage
import concurrent.futures
import uuid
import shutil

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# MCP connection details
mcp_url = "https://agents-mcp-hackathon-transcript-generator.hf.space/gradio_api/mcp/sse"

# Global variable to store tools
mcp_tools = {}

# Create a directory to store temporary files
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize tools at startup
def initialize_tools():
    global mcp_tools
    try:
        # Connect to MCP server and fetch tools
        with MCPClient({"url": mcp_url, "transport": "sse"}) as mcp_client:
            # Store tools in the global dictionary
            mcp_tools = {tool.name: tool for tool in mcp_client}
            print(f"Loaded tools: {', '.join(mcp_tools.keys())}")
        return True
    except Exception as e:
        print(f"Failed to initialize MCP tools: {e}")
        return False

# Helper: Format history for Claude
def format_history(history):
    messages = []
    for item in history:
        if isinstance(item, dict) and 'role' in item and 'content' in item:
            messages.append({"role": item["role"], "content": item["content"]})
    return messages

# File validation function
def validate_file(file):
    """Validate uploaded file type and size."""
    if file is None:
        return False, "No file uploaded"
    
    # Check file size (25MB limit)
    file_size_mb = os.path.getsize(file) / (1024 * 1024)
    if file_size_mb > 25:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds 25MB limit"
    
    # Check file extension
    valid_extensions = ['.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.flac', '.ogg', '.aac']
    file_extension = os.path.splitext(file)[1].lower()
    if file_extension not in valid_extensions:
        return False, f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
    
    return True, "File is valid"

# Process the transcription using the MCP tool
def process_transcription(audio_file, request: gr.Request):
    try:
        global mcp_tools
        
        # Check if tools are initialized
        if not mcp_tools:
            if not initialize_tools():
                return "Error: Failed to initialize MCP tools"
        
        # Get the transcription tool
        tool = mcp_tools.get("transcript_generator_transcribe_audio")
        if not tool:
            return "Error: Transcription tool not available"
        
        # Create a copy of the file with a unique name to avoid conflicts
        file_ext = os.path.splitext(audio_file)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        temp_file_path = os.path.join(TEMP_DIR, unique_filename)
        shutil.copy2(audio_file, temp_file_path)
        
        # Get the base URL from the request
        base_url = str(request.base_url)
        if base_url.endswith('/'):
            base_url = base_url[:-1]
            
        # Create a URL that points to our custom route
        file_url = f"{base_url}/audio_files/{unique_filename}"
        
        print(f"Using file URL: {file_url}")
        
        # Create a new MCP client for this specific transaction
        with MCPClient({"url": mcp_url}) as mcp_client:
            transcription_tool = next((t for t in mcp_client if t.name == "transcript_generator_transcribe_audio"), None)
            
            if not transcription_tool:
                return "Error: Transcription tool not found in MCP client"
            
            # Call the tool with the audio file URL
            result = transcription_tool(audio_file=file_url)
            return result
            
    except Exception as e:
        import traceback
        print(f"Transcription error: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"

# Main chat function (streaming, with tool usage)
def chat_with_tools(user_message, history, audio_file, request: gr.Request):
    # Add user message
    history = history or []
    messages = format_history(history)
    
    # Check if a file was uploaded
    file_message = ""
    if audio_file is not None:
        # Validate the file
        is_valid, message = validate_file(audio_file)
        if not is_valid:
            history.append(ChatMessage(
                role="assistant",
                content=message,
                metadata={"title": "❌ Error", "status": "done"}
            ))
            yield history
            return
            
        file_name = os.path.basename(audio_file)
        file_message = f"\n[Attached file: {file_name}]"
    
    # Add user message with file reference if applicable
    messages.append({"role": "user", "content": user_message + file_message})

    # 1. "Thinking" phase
    yield history + [ChatMessage(role="assistant", content="Let me think...", metadata={"title": "🧠 Thinking", "status": "pending"})]

    # 2. Tool usage phase - Process transcription if there's an audio file
    if audio_file is not None:
        try:
            # Show tool call status
            yield history + [ChatMessage(
                role="assistant", 
                content="Processing your audio file...", 
                metadata={"title": "🛠️ Tool Usage", "status": "pending"}
            )]
            
            # Process transcription 
            tool_result = process_transcription(audio_file, request)
            
            if tool_result.startswith("Error:"):
                history.append(ChatMessage(
                    role="assistant",
                    content=tool_result,
                    metadata={"title": "❌ Error", "status": "done"}
                ))
            else:
                # Add tool result to history
                history.append(ChatMessage(
                    role="assistant",
                    content=f"**Transcription Result:**\n\n{tool_result}",
                    metadata={"title": "🎙️ Transcription", "status": "done"}
                ))
            yield history
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            
            history.append(ChatMessage(
                role="assistant",
                content=f"Error processing the audio file: {str(e)}",
                metadata={"title": "❌ Error", "status": "done"}
            ))
            yield history
    # Manual transcript request without file
    elif "transcript" in user_message.lower():
        history.append(ChatMessage(
            role="assistant",
            content="Please upload an audio or video file for transcription.",
            metadata={"title": "ℹ️ Info", "status": "done"}
        ))
        yield history

    # 3. Claude response phase
    response = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=4096,
        messages=messages
    )
    history.append(ChatMessage(role="assistant", content=response.content[0].text))
    yield history

# Initialize tools at startup
initialize_tools()

# Create the Gradio interface with file upload
with gr.Blocks() as demo:
    gr.Markdown("# Claude 3.7 Sonnet Chat with Transcription Tools")
    gr.Markdown("Upload an audio or video file to generate a transcript using the MCP server.")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages", 
                label="Claude Agent",
                height=600
            )
        
        with gr.Column(scale=1):
            audio_file = gr.File(
                label="Upload Audio/Video File",
                file_types=[".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".flac", ".ogg", ".aac"],
                type="filepath"
            )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask questions or request transcript analysis...",
            show_label=False,
            container=False,
            scale=9
        )
        btn = gr.Button("Send", scale=1)
    
    # Connect components
    btn.click(chat_with_tools, [msg, chatbot, audio_file], [chatbot], queue=True)
    msg.submit(chat_with_tools, [msg, chatbot, audio_file], [chatbot], queue=True)
    
    # Clear button
    clear = gr.Button("Clear Conversation")
    clear.click(lambda: (None, None), outputs=[chatbot, audio_file])
    
    # Add a custom route to serve audio files
    @demo.app.get("/audio_files/{filename}")
    async def serve_audio_file(filename: str):
        from fastapi.responses import FileResponse
        file_path = os.path.join(TEMP_DIR, filename)
        return FileResponse(file_path)

if __name__ == "__main__":
    demo.launch()