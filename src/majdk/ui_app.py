"""Streamlit web UI for MAJD Agent Kit"""

import streamlit as st
import json
from datetime import datetime
from .agent import Agent
from .models import OpenAIModel, OllamaModel, MockModel
from .memory import Memory


def create_agent_ui() -> Agent:
    """Create agent based on UI settings"""
    model_type = st.session_state.get("model_type", "mock")
    
    if model_type == "openai":
        api_key = st.session_state.get("openai_api_key", "")
        model_name = st.session_state.get("openai_model", "gpt-3.5-turbo")
        if api_key:
            model = OpenAIModel(api_key, model_name)
        else:
            st.error("Please provide OpenAI API key")
            return None
    elif model_type == "ollama":
        model_name = st.session_state.get("ollama_model", "llama2")
        base_url = st.session_state.get("ollama_url", "http://localhost:11434")
        model = OllamaModel(model_name, base_url)
    else:  # mock
        model = MockModel()
    
    memory = Memory("streamlit_logs.jsonl")
    return Agent(model, memory)


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="MAJD Agent Kit",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 MAJD Agent Kit")
    st.markdown("*Lightweight AI Agent with Tool Support*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            ["mock", "openai", "ollama"],
            key="model_type"
        )
        
        if model_type == "openai":
            st.text_input(
                "OpenAI API Key",
                type="password",
                key="openai_api_key",
                help="Your OpenAI API key"
            )
            st.selectbox(
                "Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                key="openai_model"
            )
        elif model_type == "ollama":
            st.text_input(
                "Model Name",
                value="llama2",
                key="ollama_model"
            )
            st.text_input(
                "Base URL",
                value="http://localhost:11434",
                key="ollama_url"
            )
        
        st.markdown("---")
        if st.button("🧹 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                agent = create_agent_ui()
                if agent:
                    response = agent.run(prompt)
                    st.markdown(response)
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    # Show agent steps in expander
                    with st.expander("🔍 Agent Steps"):
                        steps = agent.memory.get_steps()
                        for step in steps[-10:]:  # Show last 10 steps
                            st.json({
                                "type": step["step_type"],
                                "content": step["content"],
                                "time": step["timestamp"]
                            })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with **MAJD Agent Kit** • "
        "[GitHub](https://github.com/your-username/majd-agent-kit) • "
        "Apache 2.0 License"
    )


def run_app():
    """Entry point for running the Streamlit app"""
    main()


if __name__ == "__main__":
    main()
