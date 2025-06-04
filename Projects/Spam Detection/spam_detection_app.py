import streamlit as st
import requests
import json
import time
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 30px;
    }
    .spam-alert {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 10px 0;
    }
    .safe-alert {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 10px 0;
    }
    .confidence-score {
        font-size: 18px;
        font-weight: bold;
        margin: 10px 0;
    }
    .email-example {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Sample email examples
SAMPLE_EMAILS = {
    "Spam Examples": {
        "Urgent Account Verification": """
Subject: URGENT: Your Account Will Be Suspended!

Dear Customer,

Your account has been flagged for suspicious activity and will be suspended within 24 hours unless you verify your identity immediately.

Click here to verify: http://fake-bank-verify.com/urgent

Failure to act now will result in permanent account closure. Don't delay!

Best regards,
Security Team
        """,
        
        "Lottery Winner": """
Subject: CONGRATULATIONS! You've Won $500,000!

Dear Lucky Winner,

You have been selected as the winner of our international lottery! You've won $500,000 USD!

To claim your prize, send us:
- Full name
- Address
- Phone number
- Bank account details

Contact: lottery.winner@fake-lottery.net

Act fast! This offer expires in 48 hours!
        """,
        
        "Fake Investment": """
Subject: Make $5000 Per Week Working From Home!

Hi there,

I made $50,000 last month with this simple trick that banks don't want you to know!

No experience needed! Just 2 hours per day!

Limited spots available. Click here: bit.ly/get-rich-quick-123

Don't miss this opportunity!
        """,
        
        "Phishing Email": """
Subject: Security Alert - Action Required

Your PayPal account has been limited due to unusual activity.

To restore full access, please verify your account information:

Username: ____________
Password: ____________
SSN: ____________

Verify now: paypal-security-check.suspicious-site.com

This is urgent. Your account will be closed if not verified within 24 hours.
        """
    },
    
    "Legitimate Examples": {
        "Work Email": """
Subject: Team Meeting Tomorrow at 2 PM

Hi everyone,

Just a reminder that we have our weekly team meeting tomorrow (Friday) at 2:00 PM in Conference Room B.

Agenda:
- Project status updates
- Q3 planning discussion
- New client onboarding process

Please bring your project reports and any questions you'd like to discuss.

Thanks,
Sarah Johnson
Project Manager
        """,
        
        "Newsletter": """
Subject: Your Weekly Tech News Digest

Hello,

Here are this week's top technology stories:

1. New smartphone releases from major manufacturers
2. Updates in artificial intelligence research
3. Cybersecurity best practices for small businesses
4. Upcoming tech conferences and events

You can read the full articles on our website. If you no longer wish to receive these emails, you can unsubscribe at the bottom of this message.

Thank you for subscribing!

Tech News Team
        """,
        
        "Order Confirmation": """
Subject: Order Confirmation #12345

Dear Customer,

Thank you for your order! Here are your order details:

Order Number: #12345
Date: March 15, 2024
Items: 
- Laptop Computer - $899.99
- Wireless Mouse - $29.99

Total: $929.98

Your order will be shipped within 2-3 business days to the address provided during checkout.

You can track your order at: account.legitimate-store.com/orders

Best regards,
Customer Service Team
        """,
        
        "Password Reset": """
Subject: Password Reset Request

Hello,

We received a request to reset your password for your account at ourservice.com.

If you made this request, please click the link below to reset your password:
https://ourservice.com/reset-password?token=abc123

This link will expire in 2 hours for security reasons.

If you did not request this password reset, please ignore this email and your password will remain unchanged.

Best regards,
Security Team
        """
    }
}

class SpamDetector:
    def __init__(self):
        self.api_endpoints = {
            "OpenAI": self.call_openai,
            "Gemini": self.call_gemini,
            "Claude": self.call_claude
        }
    
    def call_openai(self, email_content, api_key):
        """Call OpenAI API for spam detection"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""
            Analyze the following email and determine if it's spam or legitimate.
            
            Email content:
            {email_content}
            
            Please respond with a JSON object containing:
            - "classification": "spam" or "legitimate"
            - "confidence": a number between 0 and 100
            - "reasoning": brief explanation of your decision
            - "red_flags": list of suspicious elements found (if any)
            """
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are an expert email spam detector. Analyze emails and provide accurate spam detection results in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                try:
                    # Clean the JSON response
                    json_content = content.strip()
                    if json_content.startswith('```json'):
                        json_content = json_content[7:-3]
                    elif json_content.startswith('```'):
                        json_content = json_content[3:-3]
                    return json.loads(json_content)
                except:
                    return self.parse_response_text(content)
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please wait a moment and try again."}
            elif response.status_code == 401:
                return {"error": "Invalid API key. Please check your OpenAI API key."}
            elif response.status_code == 402:
                return {"error": "Insufficient credits. Please add credits to your OpenAI account."}
            else:
                return {"error": f"OpenAI API Error {response.status_code}: {response.text}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout. Please try again."}
        except Exception as e:
            return {"error": f"OpenAI API Error: {str(e)}"}
    
    def call_gemini(self, email_content, api_key):
        """Call Gemini API for spam detection"""
        try:
            prompt = f"""
            Analyze this email for spam detection:
            
            {email_content}
            
            Respond with JSON format only:
            {{
                "classification": "spam" or "legitimate",
                "confidence": number 0-100,
                "reasoning": "explanation",
                "red_flags": ["list", "of", "issues"]
            }}
            """
            
            # Updated Gemini API endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1000,
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }
            
            response = requests.post(url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    try:
                        # Clean the JSON response
                        json_content = content.strip()
                        if json_content.startswith('```json'):
                            json_content = json_content[7:-3]
                        elif json_content.startswith('```'):
                            json_content = json_content[3:-3]
                        return json.loads(json_content)
                    except:
                        return self.parse_response_text(content)
                else:
                    return {"error": "No response from Gemini API"}
            elif response.status_code == 400:
                return {"error": "Invalid request. Please check your API key and try again."}
            elif response.status_code == 403:
                return {"error": "Access forbidden. Please check your Gemini API key permissions."}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please wait a moment and try again."}
            else:
                error_details = ""
                try:
                    error_response = response.json()
                    if 'error' in error_response:
                        error_details = error_response['error'].get('message', '')
                except:
                    error_details = response.text
                return {"error": f"Gemini API Error {response.status_code}: {error_details}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout. Please try again."}
        except Exception as e:
            return {"error": f"Gemini API Error: {str(e)}"}
    
    def call_claude(self, email_content, api_key):
        """Call Claude API for spam detection"""
        try:
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            prompt = f"""
            Analyze this email to determine if it's spam or legitimate:
            
            {email_content}
            
            Please respond with ONLY a JSON object in this exact format:
            {{
                "classification": "spam" or "legitimate",
                "confidence": number between 0-100,
                "reasoning": "brief explanation",
                "red_flags": ["suspicious", "elements", "found"]
            }}
            """
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1000,
                "temperature": 0.1,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text']
                try:
                    # Clean the JSON response
                    json_content = content.strip()
                    if json_content.startswith('```json'):
                        json_content = json_content[7:-3]
                    elif json_content.startswith('```'):
                        json_content = json_content[3:-3]
                    return json.loads(json_content)
                except:
                    return self.parse_response_text(content)
            elif response.status_code == 401:
                return {"error": "Invalid API key. Please check your Claude API key."}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded. Please wait and try again."}
            elif response.status_code == 400:
                return {"error": "Invalid request format. Please try again."}
            else:
                error_details = ""
                try:
                    error_response = response.json()
                    if 'error' in error_response:
                        error_details = error_response['error'].get('message', '')
                except:
                    error_details = response.text
                return {"error": f"Claude API Error {response.status_code}: {error_details}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout. Please try again."}
        except Exception as e:
            return {"error": f"Claude API Error: {str(e)}"}
    
    def parse_response_text(self, text):
        """Parse non-JSON responses"""
        classification = "legitimate"
        confidence = 50
        reasoning = text
        red_flags = []
        
        # Simple parsing logic
        if any(word in text.lower() for word in ["spam", "suspicious", "phishing", "scam"]):
            classification = "spam"
            confidence = 75
        
        return {
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning,
            "red_flags": red_flags
        }
    
    def detect_spam(self, email_content, selected_apis, api_keys):
        """Main spam detection function"""
        results = {}
        
        for api_name in selected_apis:
            if api_name in api_keys and api_keys[api_name]:
                with st.spinner(f"Analyzing with {api_name}..."):
                    result = self.api_endpoints[api_name](email_content, api_keys[api_name])
                    results[api_name] = result
                    time.sleep(1)  # Rate limiting
        
        return results

# Initialize the detector
detector = SpamDetector()

# Main app
st.markdown("<h1 class='main-header'>📧 Email Spam Detection System</h1>", unsafe_allow_html=True)

st.markdown("""
This application uses multiple AI models (OpenAI, Gemini, Claude) to analyze emails and detect spam.
Upload your API keys and paste an email to get started!
""")

# Sidebar for API configuration
st.sidebar.header("🔑 API Configuration")

api_keys = {}
api_keys["OpenAI"] = st.sidebar.text_input("OpenAI API Key", type="password", help="Your OpenAI API key")
api_keys["Gemini"] = st.sidebar.text_input("Gemini API Key", type="password", help="Your Google Gemini API key")
api_keys["Claude"] = st.sidebar.text_input("Claude API Key", type="password", help="Your Anthropic Claude API key")

selected_apis = st.sidebar.multiselect(
    "Select AI Models to Use",
    options=["OpenAI", "Gemini", "Claude"],
    default=["OpenAI"],
    help="Choose which AI models to use for spam detection"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** You need valid API keys for the selected models.")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Email Analysis")
    
    # Email input
    email_content = st.text_area(
        "Paste your email content here:",
        height=300,
        placeholder="Paste the email content you want to analyze for spam..."
    )
    
    # Analysis button
    if st.button("🔍 Analyze Email", type="primary"):
        if not email_content.strip():
            st.error("Please enter email content to analyze.")
        elif not selected_apis:
            st.error("Please select at least one AI model.")
        elif not any(api_keys[api] for api in selected_apis):
            st.error("Please provide API keys for the selected models.")
        else:
            # Perform spam detection
            results = detector.detect_spam(email_content, selected_apis, api_keys)
            
            st.header("📊 Analysis Results")
            
            # Display results for each API
            for api_name, result in results.items():
                st.subheader(f"Results from {api_name}")
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    classification = result.get("classification", "unknown")
                    confidence = result.get("confidence", 0)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    red_flags = result.get("red_flags", [])
                    
                    # Display classification with color coding
                    if classification.lower() == "spam":
                        st.markdown(f"""
                        <div class="spam-alert">
                            <h4>🚨 SPAM DETECTED</h4>
                            <div class="confidence-score">Confidence: {confidence}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h4>✅ LEGITIMATE EMAIL</h4>
                            <div class="confidence-score">Confidence: {confidence}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show reasoning
                    st.write("**Reasoning:**", reasoning)
                    
                    # Show red flags if any
                    if red_flags:
                        st.write("**Red Flags:**")
                        for flag in red_flags:
                            st.write(f"• {flag}")
                    
                    st.markdown("---")

with col2:
    st.header("📧 Sample Emails")
    
    # Email examples
    for category, emails in SAMPLE_EMAILS.items():
        st.subheader(category)
        
        for title, content in emails.items():
            with st.expander(title):
                st.markdown(f'<div class="email-example">{content}</div>', unsafe_allow_html=True)
                if st.button(f"Use this example", key=f"use_{title}"):
                    st.session_state.selected_email = content
                    st.experimental_rerun()

# Check if an example was selected
if 'selected_email' in st.session_state:
    st.info("Example email loaded! Scroll up to see it in the text area.")
    # Clear the session state
    del st.session_state.selected_email

# Footer
st.markdown("---")
st.markdown("""
**How to get API keys:**
- **OpenAI**: Visit [platform.openai.com](https://platform.openai.com) → API Keys
- **Gemini**: Visit [makersuite.google.com](https://makersuite.google.com) → Get API Key  
- **Claude**: Visit [console.anthropic.com](https://console.anthropic.com) → API Keys

**Security Note**: Your API keys are only stored locally in your browser session and are not saved or transmitted anywhere except to the respective AI services.
""")