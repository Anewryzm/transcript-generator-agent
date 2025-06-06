import gradio as gr
import anthropic
import os
from dotenv import load_dotenv
from smolagents.mcp_client import MCPClient
from gradio import ChatMessage

# Load environment variables
load_dotenv()

# Get API keys from environment or use None as default
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Initialize Anthropic client (will be updated if user provides a key)
client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

# MCP connection details
mcp_url = "https://agents-mcp-hackathon-transcript-generator.hf.space/gradio_api/mcp/sse"

# Global variable to store tools
mcp_tools = {}

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

# Process the transcription using the MCP tool with URL
def process_transcription_from_url(audio_url, user_groq_api_key):
    try:
        global mcp_tools
        global groq_api_key
        
        # Use environment variable or user-provided key
        current_groq_api_key = groq_api_key if groq_api_key else user_groq_api_key
        
        # Check if tools are initialized
        if not mcp_tools:
            if not initialize_tools():
                return "Error: Failed to initialize MCP tools"
        
        # Get the transcription tool
        tool = mcp_tools.get("transcript_generator_transcribe_audio_from_url")
        if not tool:
            return "Error: Transcription tool not available"
            
        # Create a new MCP client for this specific transaction
        with MCPClient({"url": mcp_url}) as mcp_client:
            transcription_tool = next((t for t in mcp_client if t.name == "transcript_generator_transcribe_audio_from_url"), None)
            
            if not transcription_tool:
                return "Error: Transcription tool not found in MCP client"
            
            # Call the tool with the audio URL and GROQ API key
            result = transcription_tool(audio_url=audio_url, api_key=current_groq_api_key)
            return result
            
    except Exception as e:
        import traceback
        print(f"Transcription error: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"

# Extract URL from message
def extract_url(message):
    import re
    # Simple URL extraction regex
    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, message)
    if match:
        return match.group(0)
    return None

# Main chat function (streaming, with tool usage)
def chat_with_tools(user_message, history, user_groq_api_key, user_anthropic_api_key):
    global client, anthropic_api_key, groq_api_key
    
    # Check if we need to update the Anthropic client with user-provided API key
    current_anthropic_api_key = anthropic_api_key if anthropic_api_key else user_anthropic_api_key
    if not current_anthropic_api_key:
        history.append(ChatMessage(
            role="assistant",
            content="Please provide an Anthropic API key to continue.",
            metadata={"title": "❌ Error", "status": "done"}
        ))
        yield history
        return
    
    # Update client if needed
    if client is None or (not anthropic_api_key and user_anthropic_api_key):
        client = anthropic.Anthropic(api_key=current_anthropic_api_key)
    
    # Add user message
    history = history or []
    messages = format_history(history)
    
    # Add user message
    messages.append({"role": "user", "content": user_message})

    # 1. "Thinking" phase
    yield history + [ChatMessage(role="assistant", content="Let me think...", metadata={"title": "🧠 Thinking", "status": "pending"})]

    # 2. Tool usage phase - Process transcription if there's an audio URL
    audio_url = extract_url(user_message)
    transcript_request = "transcript" in user_message.lower() or "transcribe" in user_message.lower()
    
    # Flag to track if transcription was performed
    transcription_performed = False
    transcription_result = None
    
    if audio_url and transcript_request:
        try:
            # Check if GROQ API key is available (either from env or user input)
            current_groq_api_key = groq_api_key if groq_api_key else user_groq_api_key
            if not current_groq_api_key:
                history.append(ChatMessage(
                    role="assistant",
                    content="Please provide a GROQ API key to process the transcription.",
                    metadata={"title": "ℹ️ Info", "status": "done"}
                ))
                yield history
                return
                
            # Show tool call status
            yield history + [ChatMessage(
                role="assistant", 
                content=f"Processing audio from URL: {audio_url}...", 
                metadata={"title": "🛠️ Tool Usage", "status": "pending"}
            )]
            
            # Process transcription with GROQ API key
            transcription_result = process_transcription_from_url(audio_url, user_groq_api_key)
            
            if transcription_result.startswith("Error:"):
                history.append(ChatMessage(
                    role="assistant",
                    content=transcription_result,
                    metadata={"title": "❌ Error", "status": "done"}
                ))
                yield history
                return
            else:
                # Add transcript result to history
                history.append(ChatMessage(
                    role="assistant",
                    content=f"**Transcription Result:**\n\n{transcription_result}",
                    metadata={"title": "🎙️ Transcription", "status": "done"}
                ))
                transcription_performed = True
                yield history
                
                # Now let Claude respond to the transcript
                try:
                    # Create a new prompt for Claude to respond to the transcript
                    claude_prompt = f"You've just transcribed this audio for the user:\n\n{transcription_result}\n\nPlease respond to the user about this transcript."
                    
                    # Send request to Claude
                    response = client.messages.create(
                        model="claude-3-7-sonnet-latest",
                        max_tokens=4096,
                        system="You are a helpful assistant that can transcribe audio using MCP tools and respond to user queries. The way you can generate transcripts is using receiving audio URLs and processing them with GROQ API. When you respond to the user, always provide a clear and concise answer. You don't add unnecesary analysis or comments, unless the user asks you for it. Then you add follow up questions to the user about if they need to do something with the transcript.",
                        messages=[{"role": "user", "content": claude_prompt}]
                    )
                    
                    # Add Claude's response to history
                    response_content = response.content[0].text
                    history.append(ChatMessage(role="assistant", content=response_content))
                    yield history
                    return
                except Exception as e:
                    history.append(ChatMessage(
                        role="assistant",
                        content=f"Error generating response to transcript: {str(e)}",
                        metadata={"title": "❌ Error", "status": "done"}
                    ))
                    yield history
                    return
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            
            history.append(ChatMessage(
                role="assistant",
                content=f"Error processing the audio URL: {str(e)}",
                metadata={"title": "❌ Error", "status": "done"}
            ))
            yield history
            return
    # Manual transcript request without URL
    elif transcript_request and not audio_url:
        history.append(ChatMessage(
            role="assistant",
            content="Please provide an audio URL for transcription. I need a direct link to the audio file you want to transcribe.",
            metadata={"title": "ℹ️ Info", "status": "done"}
        ))
        yield history
        return

    # 3. Claude response phase - only for non-transcription requests
    try:
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=4096,
            messages=messages
        )
        
        response_content = response.content[0].text
        history.append(ChatMessage(role="assistant", content=response_content))
    except Exception as e:
        history.append(ChatMessage(
            role="assistant",
            content=f"Error connecting to Claude: {str(e)}",
            metadata={"title": "❌ Error", "status": "done"}
        ))
    
    yield history

# Initialize tools at startup
initialize_tools()

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Claude 3.7 Sonnet Chat with Transcription Tools")
    gr.Markdown("Send a message with an audio URL to generate a transcript using the MCP server.")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages", 
                label="Claude Agent",
                height=600
            )
        
        with gr.Column(scale=1):
            # Display API key inputs only if environment variables are not set
            anthropic_api_key_input = gr.Textbox(
                label="Anthropic API Key" + (" (using env variable)" if anthropic_api_key else ""),
                placeholder="Enter your Anthropic API key" if not anthropic_api_key else "Using environment variable",
                type="password",
                show_label=True,
                interactive=not anthropic_api_key,
                value="" if not anthropic_api_key else None
            )
            
            groq_api_key_input = gr.Textbox(
                label="GROQ API Key" + (" (using env variable)" if groq_api_key else ""),
                placeholder="Enter your GROQ API key" if not groq_api_key else "Using environment variable",
                type="password",
                show_label=True,
                interactive=not groq_api_key,
                value="" if not groq_api_key else None
            )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask questions or paste an audio URL with 'transcribe' to generate a transcript...",
            show_label=False,
            container=False,
            scale=9
        )
        btn = gr.Button("Send", scale=1)
    
    # Connect components
    btn.click(chat_with_tools, [msg, chatbot, groq_api_key_input, anthropic_api_key_input], [chatbot], queue=True)
    msg.submit(chat_with_tools, [msg, chatbot, groq_api_key_input, anthropic_api_key_input], [chatbot], queue=True)
    
    # Clear button
    clear = gr.Button("Clear Conversation")
    clear.click(lambda: (None, None, None), outputs=[chatbot, groq_api_key_input, anthropic_api_key_input])

if __name__ == "__main__":
    demo.launch()