import streamlit as st
from openai import OpenAI, OpenAIError

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_email_response(email_text, tone):
    prompt = f"""
You are an AI assistant. Write a reply to the following email using a {tone.lower()} tone:

Email:
{email_text}

Reply:
"""
    try:
        # Try GPT-4 first
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
    except OpenAIError:
        # Fallback to GPT-3.5 if GPT-4 fails (quota exceeded, etc.)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
        except OpenAIError as e2:
            st.error(f"OpenAI API error: {e2}")
            return "Error generating email response."

    return response.choices[0].message.content
