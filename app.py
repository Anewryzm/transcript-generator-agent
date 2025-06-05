import gradio as gr
import anthropic
import os
from dotenv import load_dotenv
from smolagents.mcp_client import MCPClient
from gradio import ChatMessage

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Connect to your MCP server and fetch tools
mcp_url = "https://agents-mcp-hackathon-transcript-generator.hf.space/gradio_api/mcp/sse"
tools = None
with MCPClient({"url": mcp_url}) as mcp_tools:
    tools = {tool.name: tool for tool in mcp_tools}

    # log the tools loaded
    print(f"Loaded tools: {', '.join(tools.keys())}")

# Helper: Format history for Claude
def format_history(history):
    messages = []
    for item in history:
        if isinstance(item, dict) and 'role' in item and 'content' in item:
            messages.append({"role": item["role"], "content": item["content"]})
    return messages

# Main chat function (streaming, with tool usage)
def chat_with_tools(user_message, history, audio_file):
    # Add user message
    history = history or []
    messages = format_history(history)
    
    # Check if a file was uploaded
    file_message = ""
    if audio_file is not None:
        file_name = os.path.basename(audio_file)
        file_message = f"\n[Attached file: {file_name}]"
    
    # Add user message with file reference if applicable
    messages.append({"role": "user", "content": user_message + file_message})

    # 1. "Thinking" phase
    yield history + [ChatMessage(role="assistant", content="Let me think...", metadata={"title": "🧠 Thinking", "status": "pending"})]

    # 2. Tool usage phase - Process transcription if there's an audio file
    if audio_file is not None:
        tool = tools.get("transcript_generator_transcribe_audio")
        if tool:
            try:
                # Show tool call status
                yield history + [ChatMessage(
                    role="assistant", 
                    content="Processing your audio file...", 
                    metadata={"title": "🛠️ Tool Usage", "status": "pending"}
                )]
                
                # Call the transcription tool with the uploaded file
                tool_result = tool(audio_file)
                
                # Add tool result to history
                history.append(ChatMessage(
                    role="assistant",
                    content=f"**Transcription Result:**\n\n{tool_result}",
                    metadata={"title": "🎙️ Transcription", "status": "done"}
                ))
                yield history
            except Exception as e:
                history.append(ChatMessage(
                    role="assistant",
                    content=f"Error processing the audio file: {str(e)}",
                    metadata={"title": "❌ Error", "status": "error"}
                ))
                yield history
        else:
            history.append(ChatMessage(
                role="assistant",
                content="Transcription tool not available.",
                metadata={"title": "❌ Error", "status": "error"}
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
                file_types=["audio/*", "video/*"],
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

if __name__ == "__main__":
    demo.launch()