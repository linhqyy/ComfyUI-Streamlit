import streamlit as st
import argparse
import os

from utils.api_formatter import request_inference
from utils.util import  delete_queue, image_thumbnail, get_available_loras
from utils.dialog import comparer


def side_bar():
    with st.sidebar:
        st.title("🎨 Text to Image")
        
        # Text input for the prompt
        prompt = st.text_area("Prompt", height=100)
        
        # Batch size selection
        batch_size = st.number_input("Number of images", min_value=1, max_value=4, value=1)
        
        # Seed input
        seed = st.number_input("Seed", min_value=-1, max_value=2**32-1, value=-1)
        
        # LoRA selection
        lora_models = get_available_loras()
        selected_lora = st.selectbox("Select LoRA", lora_models)
        st.session_state['selected_lora'] = selected_lora
        
        # LoRA strength slider (only show if a LoRA is selected)
        if selected_lora != "None":
            lora_strength = st.slider("LoRA Strength", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
            st.session_state['lora_strength'] = lora_strength
        
        # Generate button
        generate_clicked = st.button("Generate")
        
        return prompt, batch_size, seed, generate_clicked

def main():
    st.set_page_config(
        page_title="HG AI Team",
        page_icon="🎨",
        layout="wide"
    )
    
    
    # Initialize session state for server address
    if "server_address" not in st.session_state:
        st.session_state["server_address"] = "localhost:8188"
    
    # Create a centered column with controlled width
    col1, main_col, col2 = st.columns([1, 2, 1])  # This creates a 1:2:1 ratio layout
    
    # Get inputs from sidebar
    prompt, batch_size, seed, generate_clicked = side_bar()
    
    # Handle generation in main column
    if generate_clicked:
        if not prompt:
            st.sidebar.error("Please enter a prompt")
        else:
            with main_col:  # Use the main column instead of full width
                request_inference(
                    server_address=st.session_state["server_address"],
                    prompt=prompt,
                    batch_size=batch_size,
                    seed=seed,
                    container=main_col
                )

if __name__ == "__main__":
    main()
