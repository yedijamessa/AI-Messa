import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import numpy as np
from collections import Counter

# Installation instructions displayed if packages are missing
def show_installation_instructions():
    st.error("Missing required packages!")
    st.code("""
# Install required packages:
pip install streamlit pandas numpy plotly

# For API integration (optional):
pip install openai>=1.0.0
pip install anthropic
pip install google-generativeai

# Or install everything at once:
pip install streamlit pandas numpy plotly openai anthropic google-generativeai
    """)

# Only import what's absolutely necessary to avoid dependency issues
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Install with: `pip install plotly`")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="📄 Sentiment Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        margin: 0.5rem 0;
    }
    
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Title
st.markdown('<h1 class="main-header">📄 Document Sentiment Analyzer</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🔧 Configuration")

# Check available models
available_models = ["Demo Mode (No API)"]
if OPENAI_AVAILABLE:
    available_models.extend(["OpenAI GPT-3.5", "OpenAI GPT-4"])
if CLAUDE_AVAILABLE:
    available_models.append("Claude 3 Sonnet")
if GEMINI_AVAILABLE:
    available_models.append("Gemini Pro")

selected_model = st.sidebar.selectbox("Choose AI Model:", available_models)

# API Key input (only show if not demo mode)
api_key = None
if selected_model != "Demo Mode (No API)":
    if "OpenAI" in selected_model:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    elif "Claude" in selected_model:
        api_key = st.sidebar.text_input("Claude API Key", type="password")
    elif "Gemini" in selected_model:
        api_key = st.sidebar.text_input("Gemini API Key", type="password")

# Analysis options
st.sidebar.markdown("---")
st.sidebar.header("📊 Analysis Options")
analysis_depth = st.sidebar.selectbox("Analysis Depth:", ["Quick", "Standard", "Detailed"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)

# Mock sentiment analysis function (for demo mode)
def mock_sentiment_analysis(text, depth="Standard"):
    """Simple keyword-based sentiment analysis for demo purposes"""
    time.sleep(1)  # Simulate API delay
    
    positive_words = ['love', 'amazing', 'great', 'excellent', 'fantastic', 'wonderful', 'perfect', 'outstanding']
    negative_words = ['hate', 'terrible', 'awful', 'horrible', 'worst', 'disappointing', 'bad', 'poor']
    
    text_lower = text.lower()
    positive_score = sum(1 for word in positive_words if word in text_lower)
    negative_score = sum(1 for word in negative_words if word in text_lower)
    
    if positive_score > negative_score:
        sentiment = "positive"
        confidence = min(0.7 + positive_score * 0.1, 0.95)
    elif negative_score > positive_score:
        sentiment = "negative"
        confidence = min(0.7 + negative_score * 0.1, 0.95)
    else:
        sentiment = "neutral"
        confidence = 0.6 + np.random.uniform(-0.1, 0.1)
    
    result = {
        "sentiment": sentiment,
        "confidence": confidence,
        "summary": f"Text classified as {sentiment} with {confidence:.2f} confidence."
    }
    
    if depth in ["Standard", "Detailed"]:
        result["emotions"] = {
            "positive": ["joy", "satisfaction"],
            "negative": ["disappointment", "frustration"],
            "neutral": ["calm", "neutral"]
        }[sentiment]
        
        result["key_phrases"] = [phrase.strip() for phrase in text.split('.') if len(phrase.strip()) > 10][:3]
    
    if depth == "Detailed":
        result["intensity"] = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        result["recommendations"] = [
            f"The text shows {sentiment} sentiment. Consider this in your analysis."
        ]
    
    return result

# Real API analysis functions
def analyze_with_openai(text, model="gpt-3.5-turbo"):
    """Analyze sentiment using OpenAI"""
    if not api_key:
        st.error("Please provide OpenAI API key")
        return None
    
    try:
        # Handle both old and new OpenAI API versions
        try:
            # New API (v1.0+)
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Analyze sentiment and return JSON with: sentiment (positive/negative/neutral), confidence (0-1), summary"},
                    {"role": "user", "content": f"Analyze: {text}"}
                ],
                temperature=0.1
            )
            result_text = response.choices[0].message.content
            
        except (ImportError, AttributeError):
            # Old API (v0.x)
            openai.api_key = api_key
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Analyze sentiment and return JSON with: sentiment (positive/negative/neutral), confidence (0-1), summary"},
                    {"role": "user", "content": f"Analyze: {text}"}
                ],
                temperature=0.1
            )
            result_text = response.choices[0].message.content
        
        # Parse JSON response
        try:
            return json.loads(result_text)
        except:
            return {"sentiment": "neutral", "confidence": 0.5, "summary": result_text}
            
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        return None

def analyze_with_claude(text):
    """Analyze sentiment using Claude"""
    if not api_key:
        st.error("Please provide Claude API key")
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            messages=[{
                "role": "user", 
                "content": f"Analyze sentiment of this text and return JSON with sentiment, confidence, summary: {text}"
            }]
        )
        
        result_text = response.content[0].text
        try:
            return json.loads(result_text)
        except:
            return {"sentiment": "neutral", "confidence": 0.5, "summary": result_text}
    except Exception as e:
        st.error(f"Claude API error: {str(e)}")
        return None

def analyze_with_gemini(text):
    """Analyze sentiment using Gemini"""
    if not api_key:
        st.error("Please provide Gemini API key")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # Try different model names
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro'
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except:
                continue
        
        if not model:
            # Fallback: list available models
            try:
                available_models = list(genai.list_models())
                if available_models:
                    # Use the first available generative model
                    for m in available_models:
                        if 'generateContent' in m.supported_generation_methods:
                            model = genai.GenerativeModel(m.name)
                            break
            except:
                pass
        
        if not model:
            st.error("No compatible Gemini model found. Please check your API key.")
            return None
        
        prompt = f"""Analyze the sentiment of this text and return a JSON response with:
        - sentiment: "positive", "negative", or "neutral"
        - confidence: a number between 0 and 1
        - summary: brief explanation
        
        Text: {text}"""
        
        response = model.generate_content(prompt)
        result_text = response.text
        
        try:
            # Clean up response text
            clean_text = result_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]
            
            return json.loads(clean_text.strip())
        except:
            return {"sentiment": "neutral", "confidence": 0.5, "summary": result_text}
            
    except Exception as e:
        st.error(f"Gemini API error: {str(e)}")
        return None

# Main analysis function
def analyze_sentiment(text, model, depth):
    """Main sentiment analysis function"""
    if model == "Demo Mode (No API)":
        return mock_sentiment_analysis(text, depth)
    elif "OpenAI GPT-3.5" in model:
        return analyze_with_openai(text, "gpt-3.5-turbo")
    elif "OpenAI GPT-4" in model:
        return analyze_with_openai(text, "gpt-4")
    elif "Claude" in model:
        return analyze_with_claude(text)
    elif "Gemini" in model:
        return analyze_with_gemini(text)
    else:
        return mock_sentiment_analysis(text, depth)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Analysis", "📄 Document Upload", "📊 Batch Analysis", "📈 History"])

# Tab 1: Text Analysis
with tab1:
    st.header("📝 Single Text Analysis")
    
    # Sample texts
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 Load Positive Example"):
            st.session_state.sample_text = "I absolutely love this product! The quality is outstanding and it exceeded all my expectations. Amazing customer service!"
    with col2:
        if st.button("😞 Load Negative Example"):
            st.session_state.sample_text = "This was terrible! Poor quality and awful customer service. Complete waste of money."
    with col3:
        if st.button("😐 Load Neutral Example"):
            st.session_state.sample_text = "The product meets basic requirements. Delivery was on time. Standard quality as expected."
    
    # Text input
    input_text = st.text_area(
        "Enter text to analyze:",
        value=st.session_state.get('sample_text', ''),
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if input_text.strip():
            with st.spinner(f"Analyzing with {selected_model}..."):
                result = analyze_sentiment(input_text, selected_model, analysis_depth)
                
                if result:
                    # Display results
                    sentiment = result.get("sentiment", "unknown")
                    confidence = result.get("confidence", 0)
                    
                    # Create columns for metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sentiment", sentiment.title())
                    with col2:
                        st.metric("Confidence", f"{confidence:.3f}")
                    with col3:
                        st.metric("Text Length", len(input_text))
                    with col4:
                        st.metric("Word Count", len(input_text.split()))
                    
                    # Sentiment display
                    if sentiment == "positive":
                        st.markdown(f"""
                        <div class="sentiment-positive">
                            <h3>😊 Positive Sentiment</h3>
                            <p>Confidence: {confidence:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif sentiment == "negative":
                        st.markdown(f"""
                        <div class="sentiment-negative">
                            <h3>😞 Negative Sentiment</h3>
                            <p>Confidence: {confidence:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="sentiment-neutral">
                            <h3>😐 Neutral Sentiment</h3>
                            <p>Confidence: {confidence:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional details
                    if "emotions" in result:
                        st.write("**Detected Emotions:**", ", ".join(result["emotions"]))
                    
                    if "key_phrases" in result and result["key_phrases"]:
                        st.write("**Key Phrases:**")
                        for phrase in result["key_phrases"]:
                            st.write(f"• {phrase}")
                    
                    if "intensity" in result:
                        st.write(f"**Intensity:** {result['intensity']}")
                    
                    if "summary" in result:
                        st.info(f"**Summary:** {result['summary']}")
                    
                    if "recommendations" in result:
                        st.write("**Recommendations:**")
                        for rec in result["recommendations"]:
                            st.write(f"• {rec}")
                    
                    # Add to history
                    history_item = {
                        "timestamp": datetime.now(),
                        "text": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                        "model": selected_model,
                        "result": result
                    }
                    st.session_state.analysis_history.append(history_item)
                else:
                    st.error("Analysis failed. Please check your API key and try again.")
        else:
            st.warning("Please enter some text to analyze.")

# Tab 2: Document Upload
with tab2:
    st.header("📄 Document Analysis")
    
    uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
    
    if uploaded_file is not None:
        # Read file content
        content = uploaded_file.read().decode("utf-8")
        
        st.success(f"File uploaded successfully! ({len(content)} characters)")
        
        # Show preview
        with st.expander("📖 File Preview"):
            st.text(content[:500] + "..." if len(content) > 500 else content)
        
        # Analyze button
        if st.button("🔍 Analyze Document"):
            with st.spinner("Analyzing document..."):
                # For long documents, we might want to chunk them
                if len(content) > 3000:
                    st.info("Document is long. Analyzing first 3000 characters...")
                    content = content[:3000]
                
                result = analyze_sentiment(content, selected_model, analysis_depth)
                
                if result:
                    sentiment = result.get("sentiment", "unknown")
                    confidence = result.get("confidence", 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Document Sentiment", sentiment.title())
                        st.metric("Confidence", f"{confidence:.3f}")
                    with col2:
                        st.metric("Characters", len(content))
                        st.metric("Words", len(content.split()))
                    
                    if "summary" in result:
                        st.info(f"**Analysis Summary:** {result['summary']}")
                    
                    # Add to history
                    history_item = {
                        "timestamp": datetime.now(),
                        "text": f"Document: {uploaded_file.name}",
                        "model": selected_model,
                        "result": result,
                        "type": "document"
                    }
                    st.session_state.analysis_history.append(history_item)

# Tab 3: Batch Analysis
with tab3:
    st.header("📊 Batch Analysis")
    
    # Sample batch texts
    if st.button("📋 Load Sample Batch"):
        sample_batch = """I love this product! Amazing quality.
This is terrible. Poor customer service.
The weather is nice today.
Great experience, will buy again!
Not satisfied with the purchase."""
        st.session_state.batch_text = sample_batch
    
    batch_text = st.text_area(
        "Enter multiple texts (one per line):",
        value=st.session_state.get('batch_text', ''),
        height=200,
        placeholder="Enter each text on a new line...\nExample text 1\nExample text 2"
    )
    
    if st.button("🚀 Analyze Batch"):
        if batch_text.strip():
            texts = [text.strip() for text in batch_text.split('\n') if text.strip()]
            
            if texts:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, text in enumerate(texts):
                    status_text.text(f"Analyzing text {i+1}/{len(texts)}")
                    
                    result = analyze_sentiment(text, selected_model, "Quick")
                    
                    if result:
                        results.append({
                            'Text': text,
                            'Sentiment': result.get('sentiment', 'unknown'),
                            'Confidence': result.get('confidence', 0)
                        })
                    
                    progress_bar.progress((i + 1) / len(texts))
                
                # Display results
                if results:
                    st.success(f"Analyzed {len(results)} texts!")
                    
                    # Summary metrics
                    df = pd.DataFrame(results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Texts", len(df))
                    with col2:
                        positive_count = len(df[df['Sentiment'] == 'positive'])
                        st.metric("Positive", positive_count)
                    with col3:
                        negative_count = len(df[df['Sentiment'] == 'negative'])
                        st.metric("Negative", negative_count)
                    with col4:
                        neutral_count = len(df[df['Sentiment'] == 'neutral'])
                        st.metric("Neutral", neutral_count)
                    
                    # Charts (if plotly available)
                    if PLOTLY_AVAILABLE:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_counts = df['Sentiment'].value_counts()
                            fig_pie = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, 
                                           title="Sentiment Distribution")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            fig_bar = px.histogram(df, x='Confidence', color='Sentiment',
                                                 title="Confidence Distribution")
                            st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv,
                        "batch_sentiment_results.csv",
                        "text/csv"
                    )
                    
                    # Add to history
                    history_item = {
                        "timestamp": datetime.now(),
                        "text": f"Batch Analysis: {len(texts)} texts",
                        "model": selected_model,
                        "result": {"batch_results": results},
                        "type": "batch"
                    }
                    st.session_state.analysis_history.append(history_item)
                
                status_text.empty()
                progress_bar.empty()
            else:
                st.warning("No valid texts found.")
        else:
            st.warning("Please enter some texts to analyze.")

# Tab 4: History
with tab4:
    st.header("📈 Analysis History & Insights")
    
    if st.session_state.analysis_history:
        # Summary metrics
        total_analyses = len(st.session_state.analysis_history)
        
        # Calculate stats
        all_results = []
        for item in st.session_state.analysis_history:
            if item.get("type") == "batch":
                all_results.extend([r["Sentiment"] for r in item["result"]["batch_results"]])
            else:
                all_results.append(item["result"]["sentiment"])
        
        sentiment_counts = Counter(all_results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Positive", sentiment_counts.get('positive', 0))
        with col3:
            st.metric("Negative", sentiment_counts.get('negative', 0))
        with col4:
            st.metric("Neutral", sentiment_counts.get('neutral', 0))
        
        # Charts (if plotly available)
        if PLOTLY_AVAILABLE and sentiment_counts:
            fig = px.pie(values=list(sentiment_counts.values()), 
                        names=list(sentiment_counts.keys()),
                        title="Overall Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent history
        st.subheader("🕒 Recent Analyses")
        for i, item in enumerate(reversed(st.session_state.analysis_history[-10:])):
            with st.expander(f"Analysis {total_analyses - i}: {item['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                st.write(f"**Model:** {item['model']}")
                st.write(f"**Text:** {item['text']}")
                if item.get("type") == "batch":
                    st.write(f"**Type:** Batch Analysis")
                    st.write(f"**Results:** {len(item['result']['batch_results'])} texts processed")
                else:
                    result = item["result"]
                    st.write(f"**Sentiment:** {result['sentiment'].title()}")
                    st.write(f"**Confidence:** {result['confidence']:.3f}")
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.success("History cleared!")
            st.rerun()
        
        # Export history
        if st.button("📥 Export History"):
            history_data = []
            for item in st.session_state.analysis_history:
                if item.get("type") == "batch":
                    for result in item["result"]["batch_results"]:
                        history_data.append({
                            "Timestamp": item["timestamp"],
                            "Model": item["model"],
                            "Text": result["Text"],
                            "Sentiment": result["Sentiment"],
                            "Confidence": result["Confidence"],
                            "Type": "Batch"
                        })
                else:
                    history_data.append({
                        "Timestamp": item["timestamp"],
                        "Model": item["model"],
                        "Text": item["text"],
                        "Sentiment": item["result"]["sentiment"],
                        "Confidence": item["result"]["confidence"],
                        "Type": "Single"
                    })
            
            if history_data:
                df_history = pd.DataFrame(history_data)
                csv = df_history.to_csv(index=False)
                st.download_button(
                    "📥 Download History as CSV",
                    csv,
                    "sentiment_analysis_history.csv",
                    "text/csv"
                )
    else:
        st.info("No analysis history yet. Start analyzing some text to see insights here!")

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.header("📊 Session Stats")

if st.session_state.analysis_history:
    total = len(st.session_state.analysis_history)
    st.sidebar.metric("Analyses Done", total)
    st.sidebar.metric("Current Model", selected_model.split()[0])

# Footer info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>📄 Document Sentiment Analyzer</strong></p>
    <p>Built with Streamlit | Supports multiple AI models</p>
    <p><em>Analyze text sentiment with confidence and precision</em></p>
</div>
""", unsafe_allow_html=True)