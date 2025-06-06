import gradio as gr
import anthropic
import os
import json
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
            messages.append({"role": item["role"], "content": item["content"]})
        elif hasattr(item, 'role') and hasattr(item, 'content'):
            # Handle ChatMessage objects
            role = item.role
            content = item.content

            # Ensure valid role values for Claude API
            if role not in ["user", "assistant"]:
                role = "user" if role == "human" else "assistant"

            messages.append({"role": role, "content": content})
            print(f"Added message with role: {role}, content: {content[:50]}...")

    print(f"Formatted {len(messages)} messages for Claude")
    return messages

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

# Extract URL from message
def extract_url(message):
    import re
    # Simple URL extraction regex
    url_pattern = r'https?://[^\s]+'
    match = re.search(url_pattern, message)
    if match:
        return match.group(0)
    return None

# Function to decide if tool should be used
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
    
    # -------- Chain of Thoughts Display --------
    # 1. "Thinking" phase - keep this message in history to stack the thought process
    thinking_message = ChatMessage(
        role="assistant",
        content="I need to understand what the user is asking for and determine if I need to use my tools.",
        metadata={"title": "🧠 Thinking", "status": "pending"}
    )
    history.append(thinking_message)
    yield history

    # Extract URL from message
    audio_url = extract_url(user_message)
    
    # Update thinking process with URL detection
    if audio_url:
        thinking_message.content += f"\n\nI detected a URL in the message: {audio_url}"
        history[-1] = thinking_message
    yield history

    # 2. Decision phase - Ask Claude to decide if we should use the transcription tool
        # Create decision message only if audio URL is detected
    decision_message = ChatMessage(
                role="assistant",
        content="Determining if I should use the transcription tool for this URL...",
        metadata={"title": "🔍 Evaluating Tools", "status": "pending"}
    )
    history.append(decision_message)
    yield history

    use_tool, reasoning = should_use_transcription_tool(user_message, tool_descriptions)
    print(f"Tool decision: {use_tool}, Reasoning: {reasoning}")

    # Update decision message with the reasoning
    decision_message.content += f"\n\nDecision: {use_tool}\nReasoning: {reasoning}"
    decision_message.metadata["status"] = "done"
    history[-1] = decision_message
    yield history

    # 3. Tool usage phase - Process transcription if decided to use tool
    if use_tool:
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

            # Show tool call status - this stays in history
            tool_message = ChatMessage(
                role="assistant",
                content=f"Processing audio from URL: {audio_url}...",
                metadata={"title": "🛠️ Using Transcription Tool", "status": "pending"}
            )
            history.append(tool_message)
            yield history

            # Process transcription with GROQ API key
            transcription_result = process_transcription_from_url(audio_url, user_groq_api_key)
        
            # Update tool message with status
            if transcription_result and not transcription_result.startswith("Error:"):
                tool_message.content += f"\n\nTranscription successfully completed."
                tool_message.metadata["status"] = "done"
                history[-1] = tool_message
                yield history

            if transcription_result and transcription_result.startswith("Error:"):
                # Update tool message with error
                tool_message.content += f"\n\n{transcription_result}"
                tool_message.metadata = {"title": "❌ Error", "status": "done"}
                history[-1] = tool_message
                yield history
                return
            elif transcription_result:
                # Add transcript result to history as a separate message
                history.append(ChatMessage(
                    role="assistant",
                    content=f"**Transcription Result:**\n\n{transcription_result}",
                    metadata={"title": "🎙️ Transcription", "status": "done"}
                ))
                yield history

                # Now let Claude respond to the transcript
                try:
                    # Show thinking process for response generation
                    thinking_response = ChatMessage(
                        role="assistant",
                        content="Generating a response based on the transcription...",
                        metadata={"title": "💭 Formulating Response", "status": "pending"}
    )
                    history.append(thinking_response)
                    yield history

                    # Create a new prompt for Claude to respond to the transcript
                    claude_prompt = f"You've just transcribed this audio for the user:\n\n{transcription_result}\n\nPlease respond to the user about this transcript. Remember to add the transcript content in your response, but do not add unnecessary analysis or comments unless the user asks for it. Consider asking follow-up questions about what the user might want to do with the transcript."
                    
                    # Send request to Claude
                    response = client.messages.create(
                        model="claude-3-7-sonnet-latest",
                        max_tokens=4096,
                        system="You are a helpful assistant that can transcribe audio using MCP tools and respond to user queries. When you respond to the user about a transcript, provide a clear and concise answer. Don't add unnecessary analysis or comments unless the user asks for it. Consider adding follow-up questions about what the user might want to do with the transcript.",
                        messages=[{"role": "user", "content": claude_prompt}]
                        )
    
                    # Update thinking message as completed
                    thinking_response.content += "\n\nResponse ready!"
                    thinking_response.metadata["status"] = "done"
                    history[-1] = thinking_response
                    yield history

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
            else:
                history.append(ChatMessage(
                    role="assistant",
                    content="I wasn't able to get a transcription result. Please check the URL and try again.",
                    metadata={"title": "❌ Error", "status": "done"}
                ))
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
        else:
            # Tool decision was false, inform user
            history.append(ChatMessage(
                role="assistant",
                    content="I don't think this URL requires audio transcription. Let me respond to your message directly.",
                metadata={"title": "ℹ️ Info", "status": "done"}
            ))
            yield history
    elif "transcription" in user_message.lower() or "transcript" in user_message.lower():
        # User asking about transcription but no URL provided
        history.append(ChatMessage(
            role="assistant",
            content="I'd like to help with audio transcription, but I need a URL to the audio file. Please provide a direct link to the audio you want to transcribe.",
            metadata={"title": "ℹ️ Info", "status": "done"}
        ))
        yield history
        return

    # Update thinking status as done before moving to final response
    thinking_message.content += "\n\nAnalysis complete."
    thinking_message.metadata["status"] = "done"
    history[-1] = thinking_message
    yield history

    # 4. Claude response phase - for regular chat or when tool isn't needed
    try:
        # Show thinking process for regular response
        thinking_regular = ChatMessage(
            role="assistant",
            content="Generating a response to your question...",
            metadata={"title": "💭 Formulating Response", "status": "pending"}
    )
        history.append(thinking_regular)
        yield history

        # Provide system prompt with tools information
        system_prompt = f"""You are a helpful assistant that can transcribe audio using specialized tools.

        Available tools:
        - Audio transcription: You can convert speech in audio files to text when users provide audio URLs

        Tool usage guidelines:
        - The system has already determined that the transcription tool should not be used for this message
        - If the user asks about transcribing audio but didn't provide a URL, tell them you need an audio URL
        - For other queries, respond normally without mentioning the transcription functionality unless relevant

        When responding to general questions:
        - Provide helpful and accurate information
        - Be concise and direct
        - If the user's message suggests they might want to transcribe audio later, you can mention this capability
        """
        # Debug message format
        print(f"Sending {len(formatted_messages)} messages to Claude")
        response = client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=4096,
            system=system_prompt,
            messages=formatted_messages
        )

        # Update thinking message as completed
        thinking_regular.content += "\n\nResponse ready!"
        thinking_regular.metadata["status"] = "done"
        history[-1] = thinking_regular
        yield history

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
    gr.Markdown("# Claude 3.7 Sonnet Chat with Transcription Tools")
    gr.Markdown("Send a message with an audio URL to generate a transcript using the MCP server.")
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages", 
                label="Claude Agent",
                height=600,
                show_copy_button=True
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
            placeholder="Ask a question or share an audio URL to get a transcript...",
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

    # Reset conversation and clear message input but keep API keys
    def clear_conversation():
        return [], ""

    clear.click(
        clear_conversation,
        [],
        [chatbot, msg]  # Clear chatbot history and message input
    )

    # Add some CSS to make the chain of thought messages look better
    gr.Markdown("""
    <style>
    .message-wrap .message.assistant[data-testid] {
        max-width: 90%;
    }
    </style>
    """)

if __name__ == "__main__":
    demo.launch()