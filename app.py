import gradio as gr
import anthropic
import os
import json
import requests
import base64
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from smolagents.mcp_client import MCPClient
from gradio import ChatMessage

# Load environment variables
load_dotenv()

# Get API keys from environment or use None as default
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")
tts_api_url_base = os.environ.get("TTS_API_URL")

# Initialize Anthropic client (will be updated if user provides a key)
client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

# MCP connection details
mcp_url = "https://agents-mcp-hackathon-transcript-generator.hf.space/gradio_api/mcp/sse"

# TTS API endpoint
tts_api_url = tts_api_url_base+"/generate"

# Global variable to store tools
mcp_tools = {}
tool_descriptions = {}

# Initialize tools at startup
def initialize_tools():
    global mcp_tools, tool_descriptions
    try:
        # Connect to MCP server and fetch tools
        with MCPClient({"url": mcp_url, "transport": "sse"}) as mcp_client:
            # Store tools in the global dictionary
            mcp_tools = {tool.name: tool for tool in mcp_client}
            
            # Create tool descriptions for Claude
            for name, tool in mcp_tools.items():
                description = getattr(tool, 'description', f"Tool: {name}")
                parameters = getattr(tool, 'parameters', {})
                tool_descriptions[name] = {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            
            print(f"Loaded tools: {', '.join(mcp_tools.keys())}")
        return True
    except Exception as e:
        print(f"Failed to initialize MCP tools: {e}")
        return False

# Helper: Format history for Claude
def format_history(history):
    messages = []
    if not history:
        return messages

    print(f"History type: {type(history)}, length: {len(history)}")

    for i, item in enumerate(history):
        print(f"Item {i} type: {type(item)}")

        if isinstance(item, dict) and 'role' in item and 'content' in item:
            content = item["content"]
            # Clean base64 audio data from content before sending to Claude
            content = clean_base64_audio(content)
            messages.append({"role": item["role"], "content": content})
        elif hasattr(item, 'role') and hasattr(item, 'content'):
            # Handle ChatMessage objects
            role = item.role
            content = item.content

            # Clean base64 audio data from content before sending to Claude
            content = clean_base64_audio(content)

            # Ensure valid role values for Claude API
            if role not in ["user", "assistant"]:
                role = "user" if role == "human" else "assistant"

            messages.append({"role": role, "content": content})
            print(f"Added message with role: {role}, content: {content[:50]}...")

    print(f"Formatted {len(messages)} messages for Claude")
    return messages

# Helper function to clean base64 audio data from message content
def clean_base64_audio(content):
    if not isinstance(content, str):
        return content
    
    # Check if there's base64 audio data in the content
    if "<audio controls src=\"data:audio/wav;base64," in content:
        # Replace the entire audio tag and base64 data with a placeholder
        import re
        cleaned_content = re.sub(
            r'<audio controls src="data:audio/wav;base64,[^"]*"[^>]*></audio>',
            '<audio>[BASE64_AUDIO_REMOVED_TO_SAVE_CONTEXT]</audio>',
            content
        )
        
        # Log the size reduction
        size_before = len(content)
        size_after = len(cleaned_content)
        print(f"Cleaned base64 audio data. Size before: {size_before}, after: {size_after}, saved: {size_before - size_after} bytes")
        
        return cleaned_content
    
    return content

# Process the transcription using the MCP tool with URL
def process_transcription_from_url(audio_url, user_groq_api_key):
    try:
        global mcp_tools
        global groq_api_key
        
        # Use environment variable or user-provided key
        current_groq_api_key = groq_api_key if groq_api_key else user_groq_api_key
        
        if not current_groq_api_key:
            return "Error: GROQ API key is required for transcription"

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
            
            print(f"Calling transcription tool with URL: {audio_url} and API key (length): {len(current_groq_api_key)}")

            # Call the tool with the audio URL and GROQ API key
            result = transcription_tool(audio_url=audio_url, api_key=current_groq_api_key)

            print(f"Transcription result received (length): {len(result) if result else 0}")
            return result
    except Exception as e:
        import traceback
        print(f"Transcription error: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"

# Generate speech from text using the TTS API
def generate_speech_from_text(text):
    try:
        # Prepare the request data
        payload = {"prompt": text}
        headers = {"Content-Type": "application/json"}
        
        print(f"Sending TTS request for text (length): {len(text)}")
        
        # Make the POST request to the TTS API
        response = requests.post(tts_api_url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"TTS response received. Content-Type: {response.headers.get('Content-Type')}")
            
            return response.content  # Return the raw audio bytes
        else:
            return f"Error: API returned status code {response.status_code} - {response.text}"
    except Exception as e:
        import traceback
        print(f"TTS error: {e}")
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

# Function to decide if transcription tool should be used
def should_use_transcription_tool(prompt, available_tools):
    try:
        if not client:
            print("No Claude client available for tool decision")
            # Default to using tool if URL is detected, since we can't ask Claude
            return True, "Default to using tool due to URL detection"

        system_prompt = """You are an AI assistant deciding whether to use available tools. 
            You have access to a transcription tool that can transcribe audio from a URL.

            You should ONLY respond with a pure JSON object that includes:
            1. "use_tool": true/false - whether to use the transcription tool
            2. "reasoning": brief explanation of your decision

            It should start with a '{' and end with a '}'.

            Respond with true ONLY if:
            - The user is clearly asking for audio transcription
            - OR the user shared an audio URL and wants to know what's in it
            - OR the user is asking about content from an audio file

            Respond with false if:
            - The user is just chatting or asking questions not related to audio transcription
            - No audio URL is mentioned (unless the user is explicitly asking how to use the transcription feature)
            """

        # Add information about available tools
        tool_info = ""
        for tool_name, tool_data in available_tools.items():
            tool_info += f"Tool: {tool_name}\n"
            tool_info += f"Description: {tool_data.get('description', 'No description')}\n"
            tool_info += f"Parameters: {json.dumps(tool_data.get('parameters', {}))}\n\n"
        
        system_prompt += f"\n\nAvailable tools:\n{tool_info}"

        # Ensure we're sending a valid message
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": f"User message: {prompt}"}]
        )
        
        response_content = response.content[0].text
        print(f"Tool decision response: {response_content}")
        
        # Try to parse the JSON response
        try:
            result = json.loads(response_content)
            return result.get("use_tool", False), result.get("reasoning", "No reasoning provided")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response_content}")
            # Fallback: check if the response contains "true"
            return "true" in response_content.lower(), "Failed to parse JSON response"

    except Exception as e:
        import traceback
        print(f"Error in tool decision: {e}")
        print(traceback.format_exc())
        # Default to not using tool if there's an error
        return False, f"Error: {str(e)}"

# Function to decide if TTS tool should be used
def should_use_tts_tool(prompt, history=None):
    try:
        if not client:
            print("No Claude client available for TTS tool decision")
            # Default to not using tool since we can't ask Claude
            return False, "No Claude client available for decision", ""

        system_prompt = """You are an AI assistant deciding whether to use a text-to-speech (TTS) tool. 
            You have access to a TTS tool that can generate speech audio from text.

            You should ONLY respond with a pure JSON object that includes:
            1. "use_tool": true/false - whether to use the TTS tool
            2. "reasoning": brief explanation of your decision
            3. "text_to_convert": If use_tool is true, include the text that should be converted to speech. 
               Extract this from the user's message or generate appropriate text based on their request.

            It should start with a '{' and end with a '}'.

            Respond with true ONLY if:
            - The user is clearly asking to generate speech or audio from text
            - OR the user wants text read aloud
            - OR the user wants to create a voiceover or narration
            - OR the user explicitly mentions text-to-speech or TTS
            
            When deciding what text to convert:
            - If the user provides specific text in quotes or after phrases like "convert this to speech:", use that exact text
            - If the user asks to convert their previous message, use that as the text
            - If the user refers to content from the conversation history, extract that content
            - If the user asks to convert the transcript or other previous content, find that in the history
            - If the user doesn't specify text but clearly wants TTS, ask them to provide the text

            Respond with false if:
            - The user is just chatting or asking questions not related to speech generation
            - The user is asking about transcription (speech-to-text) instead
            - The user is asking about content generation
            """

        # Prepare the messages for Claude
        messages = []
        if history:
            # Include history in the context to allow referencing previous messages
            formatted_history = format_history(history)
            messages = formatted_history
            # Add the user's current prompt as the final message
            messages.append({"role": "user", "content": prompt})
        else:
            # If no history, just use the current prompt
            messages = [{"role": "user", "content": f"User message: {prompt}"}]

        # Ensure we're sending a valid message
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=500,
            system=system_prompt,
            messages=messages
        )
        
        response_content = response.content[0].text
        print(f"TTS tool decision response: {response_content}")
        
        # Try to parse the JSON response
        try:
            result = json.loads(response_content)
            text_to_convert = result.get("text_to_convert", "")
            return result.get("use_tool", False), result.get("reasoning", "No reasoning provided"), text_to_convert
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response_content}")
            # Fallback: check if the response contains "true"
            return "true" in response_content.lower(), "Failed to parse JSON response", ""

    except Exception as e:
        import traceback
        print(f"Error in TTS tool decision: {e}")
        print(traceback.format_exc())
        # Default to not using tool if there's an error
        return False, f"Error: {str(e)}", ""

# Main chat function (streaming, with tool usage)
def chat_with_tools(user_message, history, user_groq_api_key, user_anthropic_api_key):
    global client, anthropic_api_key, groq_api_key, tool_descriptions
    
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
    
    # Add user message to history for display
    history = history or []
    history.append(ChatMessage(role="user", content=user_message))
    
    # Format messages for Claude API
    formatted_messages = format_history(history)
    
    # 1. "Thinking" phase
    yield history + [ChatMessage(role="assistant", content="Let me think...", metadata={"title": "🧠 Thinking", "status": "pending"})]

    # Extract URL from message for transcription
    audio_url = extract_url(user_message)
    
    # 2. Decision phase - First check if we should use the TTS tool
    # Pass the full history to the TTS decision function
    use_tts, tts_reasoning, text_to_convert = should_use_tts_tool(user_message, history[:-1])  # Exclude the current message
    
    # 3a. Tool usage phase - Process TTS if decided to use that tool
    if use_tts:
        try:
            # Show tool call status
            yield history + [ChatMessage(
                role="assistant",
                content=f"Generating speech from text...",
                metadata={"title": "🔊 Text-to-Speech", "status": "pending"}
            )]
            
            # If no text was provided for conversion, ask the user
            if not text_to_convert:
                history.append(ChatMessage(
                    role="assistant",
                    content="I'd be happy to generate speech audio for you! Could you please provide the text you want me to convert to speech?",
                    metadata={"title": "ℹ️ Info", "status": "done"}
                ))
                yield history
                return
            
            # Process text-to-speech
            audio_data = generate_speech_from_text(text_to_convert)
            
            if isinstance(audio_data, str) and audio_data.startswith("Error:"):
                history.append(ChatMessage(
                    role="assistant",
                    content=audio_data,
                    metadata={"title": "❌ Error", "status": "done"}
                ))
                yield history
                return
            elif audio_data:
                # Convert audio data to base64 for embedding directly in HTML
                base64_audio = base64.b64encode(audio_data).decode('utf-8')
                
                # Create a response with the audio player using data URL
                response_with_audio = f"""I've converted your text to speech. Here's the audio:

    <audio controls src="data:audio/wav;base64,{base64_audio}" style="width: 100%;"></audio>

    The text that was converted: "{text_to_convert}"

    How does the audio sound? Let me know if you'd like to generate speech from any other text.
    """
                # Store a custom attribute on the message to indicate it has audio
                # This will be displayed in the UI but the base64 data will be stripped
                # when sending to Claude
                history.append(ChatMessage(
                    role="assistant", 
                    content=response_with_audio
                ))
                
                yield history
                return
            else:
                history.append(ChatMessage(
                    role="assistant",
                    content="I wasn't able to generate speech from your text. Please try again with different text.",
                    metadata={"title": "❌ Error", "status": "done"}
                ))
                yield history
                return
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"TTS error details: {error_details}")
            
            history.append(ChatMessage(
                role="assistant",
                content=f"Error generating speech: {str(e)}",
                metadata={"title": "❌ Error", "status": "done"}
            ))
            yield history
            return
    
    # 2b. If not using TTS, check if we should use the transcription tool
    if not use_tts and audio_url:
        use_transcription, transcription_reasoning = should_use_transcription_tool(user_message, tool_descriptions)
        print(f"Tool decision: {use_transcription}, Reasoning: {transcription_reasoning}")
    else:
        use_transcription = False
    
    # 3b. Process transcription if decided to use that tool
    if use_transcription and audio_url:
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
        
            if transcription_result and transcription_result.startswith("Error:"):
                history.append(ChatMessage(
                    role="assistant",
                    content=transcription_result,
                    metadata={"title": "❌ Error", "status": "done"}
                ))
                yield history
                return
            elif transcription_result:
                # Add transcript result to history with direct response
                history.append(ChatMessage(
                    role="assistant",
                    content=f"I've transcribed the audio for you. Here's what I found:\n\n{transcription_result}\n\nIs there anything specific you'd like to know about this transcript?"
                ))
                yield history
                return
            else:
                history.append(ChatMessage(
                    role="assistant",
                    content="I wasn't able to get a transcription result. Please check the URL and try again.",
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
    elif use_transcription and not audio_url:
        # User wants transcription but didn't provide URL
        history.append(ChatMessage(
            role="assistant",
            content="I'd like to help with audio transcription, but I need a URL to the audio file. Please provide a direct link to the audio you want to transcribe.",
            metadata={"title": "ℹ️ Info", "status": "done"}
        ))
        yield history
        return

        # 4. Claude response phase - for regular chat or when tool isn't needed
    try:
        # Provide system prompt with tools information
        system_prompt = f"""You are a helpful assistant that can transcribe audio and generate speech from text using specialized tools.

        Available tools:
        - Audio transcription: You can convert speech in audio files to text when users provide audio URLs
        - Text-to-Speech: You can convert text to speech audio when users want to generate spoken content

        Tool usage guidelines:
        - The system has already determined that neither the transcription nor TTS tools should be used for this message
        - If the user asks about transcribing audio but didn't provide a URL, tell them you need an audio URL
        - If the user asks about generating speech but didn't provide text, tell them you need the text to convert
        - For other queries, respond normally

        When responding to general questions:
        - Provide helpful and accurate information
        - Be concise and direct
        - If the user's message suggests they might want to use either tool later, you can mention these capabilities
        """
        # Debug message format
        print(f"Sending {len(formatted_messages)} messages to Claude")
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=4096,
            system=system_prompt,
            messages=formatted_messages
        )

        response_content = response.content[0].text
        history.append(ChatMessage(role="assistant", content=response_content))
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Claude error details: {error_details}")

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
    gr.Markdown("# Claude 3.7 Sonnet Chat with Tools")
    gr.Markdown("Send a message with an audio URL to generate a transcript, or ask to generate speech from text.")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages", 
                label="Claude Agent",
                height=600,
                render=True  # Enable HTML rendering for audio player
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
            placeholder="Ask a question, share an audio URL for transcription, or request text-to-speech...",
            show_label=False,
            container=False,
            scale=9
        )
        btn = gr.Button("Send", scale=1)
    
    # Connect components
    btn.click(chat_with_tools, [msg, chatbot, groq_api_key_input, anthropic_api_key_input], [chatbot], queue=True)
    msg.submit(chat_with_tools, [msg, chatbot, groq_api_key_input, anthropic_api_key_input], [chatbot], queue=True)

    # Clear message input field after sending
    def clear_message_input():
        return ""
    msg.submit(clear_message_input, [], [msg], queue=False)
    
    # Clear button
    clear = gr.Button("Clear Conversation")

    # Reset conversation and clear message input but keep API keys
    def clear_conversation():
        return [], ""

    clear.click(
        clear_conversation,
        [],
        [chatbot, msg]  # Clear chatbot history and message input
    )

if __name__ == "__main__":
    demo.launch()