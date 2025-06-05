import gradio as gr
import anthropic
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Define a function to handle conversations with Claude
def respond(message, history):
    # Format conversation history for Claude
    messages = []
    
    # Debug: Print the structure of history
    print(f"History: {history}")
    
    # Add chat history - clean up message format to only include role and content
    if history:
        for item in history:
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                # Only include role and content fields that Claude API expects
                messages.append({
                    "role": item["role"],
                    "content": item["content"]
                })
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
    print(f"Messages to send: {messages}")
    
    # Call the Claude API
    response = client.messages.create(
        model="claude-3-7-sonnet-latest",
        max_tokens=4096,
        messages=messages
    )
    
    return response.content[0].text

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Claude 3.7 Sonnet Chat",
    description="Chat with Anthropic's Claude 3.7 Sonnet model"
)

if __name__ == "__main__":
    demo.launch()