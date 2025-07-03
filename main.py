import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "store" not in st.session_state:
    st.session_state.store = {}

if "current_session" not in st.session_state:
    st.session_state.current_session = "chat1"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def get_model():
    """Initialize and cache the model"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("❌ Please add your GROQ_API_KEY to the .env file")
        st.stop()
    
    try:
        model = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=groq_api_key
        )
        return model
    except Exception as e:
        st.error(f"❌ Error initializing model: {str(e)}")
        st.stop()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create session history"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def create_chain(model, language="English"):
    """Create the conversation chain"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    chain = prompt | model
    return RunnableWithMessageHistory(chain, get_session_history)

def display_messages():
    """Display chat messages"""
    session_history = get_session_history(st.session_state.current_session)
    
    if len(session_history.messages) == 0:
        st.info("👋 Start a conversation by typing a message below!")
        return
    
    for message in session_history.messages:
        if isinstance(message, HumanMessage):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message.content}
            </div>
            """, unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message.content}
            </div>
            """, unsafe_allow_html=True)

def main():
    st.title("🤖 AI Chatbot Assistant")
    st.markdown("---")
    
    # Initialize model
    model = get_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Language selection
        language = st.selectbox(
            "🌐 Select Language",
            ["English", "Urdu", "Spanish", "French", "German"],
            index=0
        )
        
        st.markdown("---")
        
        # Session management
        st.subheader("💬 Sessions")
        
        # Current session
        st.write(f"**Current:** `{st.session_state.current_session}`")
        
        # Session buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New", use_container_width=True):
                import time
                new_session = f"chat_{int(time.time())}"
                st.session_state.current_session = new_session
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                if st.session_state.current_session in st.session_state.store:
                    st.session_state.store[st.session_state.current_session].clear()
                st.rerun()
        
        # Session selector
        if st.session_state.store:
            sessions = list(st.session_state.store.keys())
            if sessions:
                selected = st.selectbox(
                    "Switch Session",
                    sessions,
                    index=sessions.index(st.session_state.current_session) if st.session_state.current_session in sessions else 0
                )
                if selected != st.session_state.current_session:
                    st.session_state.current_session = selected
                    st.rerun()
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Stats")
        total_sessions = len(st.session_state.store)
        current_messages = len(get_session_history(st.session_state.current_session).messages)
        
        st.metric("Sessions", total_sessions)
        st.metric("Messages", current_messages)
    
    # Main chat area
    st.subheader(f"💬 Session: {st.session_state.current_session}")
    
    # Display messages
    messages_container = st.container()
    with messages_container:
        display_messages()
    
    # Chat input
    st.markdown("---")
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "💭 Your message:",
            height=100,
            placeholder="Type your message here...",
            help="Press Ctrl+Enter to send"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process input
    if submit and user_input.strip():
        # Create chain
        chain = create_chain(model, language)
        config = {"configurable": {"session_id": st.session_state.current_session}}
        
        # Show spinner
        with st.spinner("🤔 Thinking..."):
            try:
                # Send message
                response = chain.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )
                
                # Success message
                st.success("✅ Message sent!")
                
                # Auto-rerun to show new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.error("Please check your GROQ_API_KEY and internet connection.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
        "Powered by LangChain + Groq + Streamlit<br>"
        "Make sure your .env file contains GROQ_API_KEY"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()