import streamlit as st
import openai
import google.generativeai as genai
import anthropic
from typing import Generator
import time

# Page configuration
st.set_page_config(
    page_title="Multi-LLM Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2E86AB;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #2E86AB;
    }
    .assistant-message {
        background-color: #ffffff;
        border-left-color: #28a745;
    }
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "OpenAI GPT-3.5"

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model selection
    model_option = st.selectbox(
        "Choose LLM Provider:",
        ["Google Gemini", "Claude", "OpenAI GPT-3.5", "OpenAI GPT-4"]
    )
    st.session_state.selected_model = model_option
    
    # API Key inputs
    st.subheader("API Keys")
    
    if "openai" in model_option.lower():
        openai_key = st.text_input("OpenAI API Key:", type="password", key="openai_key")
        if openai_key:
            openai.api_key = openai_key
    
    elif "gemini" in model_option.lower():
        gemini_key = st.text_input("Google Gemini API Key:", type="password", key="gemini_key")
        if gemini_key:
            genai.configure(api_key=gemini_key)
    
    elif "claude" in model_option.lower():
        claude_key = st.text_input("Anthropic Claude API Key:", type="password", key="claude_key")
    
    # Chat settings
    st.subheader("Chat Settings")
    temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens:", 100, 4000, 1000, 100)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.markdown("<h1 class='main-header'>🤖 Multi-LLM Chatbot</h1>", unsafe_allow_html=True)

# Display current model
st.info(f"Currently using: **{st.session_state.selected_model}**")

# Function to get response from OpenAI
def get_openai_response(messages, model="gpt-3.5-turbo"):
    try:
        client = openai.OpenAI(api_key=st.session_state.get("openai_key"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except openai.RateLimitError:
        yield "❌ **OpenAI Error**: You've exceeded your API quota. Please check your billing details or try another model."
    except openai.AuthenticationError:
        yield "❌ **OpenAI Error**: Invalid API key. Please check your API key."
    except Exception as e:
        yield f"❌ **OpenAI Error**: {str(e)}"

# Function to get response from Gemini
def get_gemini_response(messages):
    try:
        # Try different model names
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        else:
            yield "❌ **Gemini Error**: No available models found. Please check your API key and model availability."
            return
        
        # Convert messages to Gemini format
        conversation_text = ""
        for msg in messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n"
        
        response = model.generate_content(
            conversation_text,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        if response.text:
            # Stream the response word by word for better UX
            words = response.text.split()
            for word in words:
                yield word + " "
                time.sleep(0.05)  # Small delay for streaming effect
        else:
            yield "❌ **Gemini Error**: No response generated. The content might have been blocked."
            
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            yield "❌ **Gemini Error**: API quota exceeded. Please check your billing or try again later."
        elif "api key" in error_msg.lower():
            yield "❌ **Gemini Error**: Invalid API key. Please check your API key."
        elif "not found" in error_msg.lower():
            yield "❌ **Gemini Error**: Model not available. Please check your API access."
        else:
            yield f"❌ **Gemini Error**: {error_msg}"

# Function to get response from Claude
def get_claude_response(messages):
    try:
        client = anthropic.Anthropic(api_key=st.session_state.get("claude_key"))
        
        # Convert messages to Claude format
        claude_messages = []
        for msg in messages:
            if msg["role"] != "system":
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Updated to latest model
            max_tokens=max_tokens,
            temperature=temperature,
            messages=claude_messages
        )
        
        # Stream the response word by word
        if response.content[0].text:
            words = response.content[0].text.split()
            for word in words:
                yield word + " "
                time.sleep(0.05)
        else:
            yield "❌ **Claude Error**: No response generated."
            
    except anthropic.RateLimitError:
        yield "❌ **Claude Error**: API rate limit exceeded. Please try again later."
    except anthropic.AuthenticationError:
        yield "❌ **Claude Error**: Invalid API key. Please check your API key."
    except Exception as e:
        yield f"❌ **Claude Error**: {str(e)}"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Prepare messages for API call
        messages_for_api = [{"role": m["role"], "content": m["content"]} 
                           for m in st.session_state.messages]
        
        # Get response based on selected model
        try:
            if "gemini" in st.session_state.selected_model.lower():
                if not st.session_state.get("gemini_key"):
                    st.error("Please enter your Google Gemini API key in the sidebar.")
                    st.stop()
                response_generator = get_gemini_response(messages_for_api)
                
            elif "claude" in st.session_state.selected_model.lower():
                if not st.session_state.get("claude_key"):
                    st.error("Please enter your Anthropic Claude API key in the sidebar.")
                    st.stop()
                response_generator = get_claude_response(messages_for_api)
                
            elif "gpt-3.5" in st.session_state.selected_model.lower():
                if not st.session_state.get("openai_key"):
                    st.error("Please enter your OpenAI API key in the sidebar.")
                    st.stop()
                response_generator = get_openai_response(messages_for_api, "gpt-3.5-turbo")
                
            elif "gpt-4" in st.session_state.selected_model.lower():
                if not st.session_state.get("openai_key"):
                    st.error("Please enter your OpenAI API key in the sidebar.")
                    st.stop()
                response_generator = get_openai_response(messages_for_api, "gpt-4")
            
            # Stream the response
            for chunk in response_generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = f"Sorry, I encountered an error: {str(e)}"
    
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Multi-LLM Chatbot | Supports OpenAI GPT, Google Gemini, and Anthropic Claude</p>
        <p>Made with ❤️ using Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Usage instructions in expander
with st.expander("📋 How to Use"):
    st.markdown("""
    **Setup Instructions:**
    1. **Get API Keys:**
       - OpenAI: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
       - Google Gemini: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
       - Anthropic Claude: Visit [Anthropic Console](https://console.anthropic.com/)
    
    2. **Configure the Chatbot:**
       - Select your preferred LLM provider from the sidebar
       - Enter the corresponding API key
       - Adjust temperature and max tokens as needed
    
    3. **Start Chatting:**
       - Type your message in the chat input at the bottom
       - The bot will respond using your selected LLM
       - Switch between models anytime using the sidebar
    
    **Features:**
    - Support for multiple LLM providers
    - Streaming responses for better user experience
    - Persistent chat history during session
    - Customizable parameters (temperature, max tokens)
    - Clean and responsive interface
    
    **Tips:**
    - Lower temperature (0.1-0.3) for more focused responses
    - Higher temperature (0.7-1.0) for more creative responses
    - Adjust max tokens based on desired response length
    """)