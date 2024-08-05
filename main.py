import streamlit as st
import requests

# Streamlit app title
st.title('Chatbot Interface')

# Function to call Flask API
def get_response(question):
    url = 'http://127.0.0.1:5000/query'  # Replace with your Flask API endpoint
    response = requests.post(url, json={'question': question})
    return response.json().get('response', 'No response from server.')

# Initialize session state to store chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Input for user question
question = st.text_input("Ask a question:")

# Show response when the button is clicked
if st.button('Submit'):
    if question:
        response = get_response(question)
        st.session_state.history.append(('User', question))
        st.session_state.history.append(('Bot', response))
    else:
        st.write("Please enter a question.")

# Custom CSS for chat interface
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    max-width: 700px;
    margin: 10px auto;
}
.chat-message {
    display: flex;
    margin-bottom: 10px;
}
.chat-message .message {
    padding: 10px;
    border-radius: 10px;
    max-width: 60%;
}
.chat-message.user .message {
    background-color: #ff5733;
    margin-left: auto;
}
.chat-message.bot .message {
    background-color: #ff5733;
    margin-right: auto;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# Display chat history
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in st.session_state.history:
    if sender == 'User':
        st.markdown(f"<div class='chat-message user'><div class='message'>{message}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message bot'><div class='message'>{message}</div></div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
