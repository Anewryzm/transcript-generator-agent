---
title: Transcript Generator Agent
author: Enrique Cardoza
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
short_description: A transcript generator agent powered by MCP servers
tags:
  - transcript
  - agent
  - agent-demo-track
  - anthropic
  - groq
  - whisper
  - smolagents
---

# Claude 3.7 Sonnet Chat with Transcription Tools

A powerful conversational agent that combines Claude 3.7 Sonnet's capabilities with specialized audio transcription tools. Built for the Gradio Agents & MCP Hackathon 2025 and hosted on Hugging Face Spaces.

## Overview

This application integrates Claude 3.7 Sonnet with the Model Context Protocol (MCP) to enable audio transcription directly within a chat interface. Users can share audio URLs during conversation, and the agent will automatically transcribe the content using GROQ's transcription capabilities.

## Features

- **Conversational AI**: Powered by Anthropic's Claude 3.7 Sonnet model
- **Audio Transcription**: Transcribe audio from URLs shared in the chat
- **Smart Tool Selection**: Automatically determines when to use transcription tools
- **Streaming Responses**: Real-time feedback during processing
- **API Key Management**: Supports both environment variables and user-provided API keys

## How It Works

### Architecture

1. **Gradio Interface**: Provides a user-friendly chat UI
2. **MCP Integration**: Connects to external tools via Model Context Protocol
3. **Tool Decision Logic**: Uses Claude to decide whether to invoke transcription
4. **Multi-phase Response**: Shows "thinking", tool usage, and final response phases

### Process Flow

1. **User Input**: User sends a message with an audio URL
2. **Decision Making**: The agent analyzes the message to determine if transcription is needed
3. **Transcription Processing**: If appropriate, the audio is sent to the transcription service
4. **Response Generation**: Claude responds to either the transcription or the original query

### MCP Integration

The application connects to an MCP server at `https://agents-mcp-hackathon-transcript-generator.hf.space/gradio_api/mcp/sse` that provides access to specialized tools. This implementation uses the `transcript_generator_transcribe_audio_from_url` tool to process audio content.

## Requirements

- **Anthropic API Key**: Required for Claude 3.7 Sonnet access
- **GROQ API Key**: Required for audio transcription functionality

## Usage Examples

### Audio Transcription

1. Share an audio URL in the chat
2. The agent will automatically detect the URL and transcribe the content
3. You'll receive both the transcription and Claude's analysis

```
please, generate a transcription for this file: https://huggingface.co/spaces/anewryzm/transcript-generator-client/resolve/main/test_files/this%20people%203.m4a
```

### General Conversation

Simply chat with the agent as you would with Claude. The agent will only use transcription tools when appropriate.

## Technical Details

### Components

- **Gradio UI**: Creates a responsive chat interface
- **Anthropic Client**: Handles communication with Claude API
- **MCPClient**: Connects to the Model Context Protocol server
- **Tool Processing Logic**: Manages the transcription workflow

### Environment Variables

The application supports setting API keys via environment variables:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `GROQ_API_KEY`: Your GROQ API key

If these are not provided as environment variables, the UI will prompt for them.

### Tool Decision Making

The agent uses Claude itself to determine whether to use the transcription tool by analyzing:
- Whether the message contains an audio URL
- The context and intent of the user's request
- The relevance of transcription to the current conversation

## Deployment

This agent is hosted on Hugging Face Spaces, making it easily accessible via web browser without any installation requirements.

## Limitations

- Only processes audio from publicly accessible URLs
- Requires valid API keys for both Anthropic and GROQ
- Transcription quality depends on the audio quality and GROQ's capabilities

## Future Improvements

- Support for more audio formats and sources
- Additional text processing tools for transcript analysis
- Integration with more LLM providers

## About MCP (Model Context Protocol)

MCP is a standardized way for language models to access external tools and capabilities. It allows models to:
- Discover available tools
- Understand tool parameters and requirements
- Invoke tools with appropriate inputs
- Process tool outputs in context