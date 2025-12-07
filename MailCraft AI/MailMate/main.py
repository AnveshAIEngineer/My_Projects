import streamlit as st
from agents.email_agent import generate_email_response
from utils.email_sender import send_email

st.set_page_config(page_title="Auto Email Responder", layout="wide")
st.title("📧 MailCraft AI – Smart replies made simple")

email_text = st.text_area("Paste the email content you received:", height=300)
recipient_email = st.text_input("Recipient Email Address")
tone = st.selectbox("Select response tone", ["Professional", "Friendly", "Apologetic", "Persuasive"])

if st.button("Generate & Send Email"):
    if not recipient_email:
        st.warning("Please enter the recipient's email address.")
    else:
        with st.spinner("Generating and sending email..."):
            response = generate_email_response(email_text, tone)
            send_status = send_email(recipient_email, response)
            st.subheader("✉️ Response")
            st.markdown(response, unsafe_allow_html=True)
            if send_status:
                st.success(f"Email sent successfully to {recipient_email}")
            else:
                st.error("Failed to send the email.")
