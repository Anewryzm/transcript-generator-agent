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
    
    # Add chat history
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:  # Check if there's an assistant response
            messages.append({"role": "assistant", "content": h[1]})
    
    # Add the current message
    messages.append({"role": "user", "content": message})
    
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