import streamlit as st
import websocket
import json
import uuid
import threading
import queue
import numpy as np
import cv2
import time
import requests
import os
from PIL import Image
from io import BytesIO

from utils.util import update_workflow, queue_workflow, receive_images, image_thumbnail

# Cache for LoRA models
_lora_models_cache = None

# Cache for workflows
_workflow_cache = None

def get_available_loras():
    global _lora_models_cache
    
    # Return cached models if available
    if _lora_models_cache is not None:
        return _lora_models_cache
        
    try:
        # Fetch models from server
        response = requests.get(f"http://{st.session_state['server_address']}/object_info")
        if response.status_code == 200:
            object_info = response.json()
            loras = ["None"]  # Default option
            
            if "LoraLoader" in object_info:
                lora_info = object_info["LoraLoader"]
                if "input" in lora_info and "required" in lora_info["input"]:
                    loras.extend(lora_info["input"]["required"]["lora_name"][0])
            
            # Cache the results
            _lora_models_cache = loras
            print(f"Debug: Found {len(loras)} LoRA models")
            return loras
            
    except Exception as e:
        print(f"Error fetching LoRA models: {str(e)}")
        return ["None"]  # Return default on error

def request_inference(server_address, prompt, batch_size, seed, container):
    client_id = str(uuid.uuid4())
    st.session_state["client_id"] = client_id
    
    workflow = load_workflow()
    
    with container:
        progress_text = st.empty()
        progress_bar = st.progress(0)
    
    image_queue = queue.Queue()
    progress_queue = queue.Queue()
    
    ws = None
    try:
        ws = websocket.WebSocket()
        ws_url = f"ws://{server_address}/ws?clientId={client_id}"
        ws.connect(ws_url)
        ws.settimeout(1)
        
        prompt_id = queue_workflow(
            workflow=workflow,
            server_address=server_address,
            client_id=client_id,
            prompt=prompt,
            batch_size=batch_size,
            seed=seed
        )
        
        thread = threading.Thread(
            target=receive_images,
            args=(ws, prompt_id, image_queue, progress_queue, batch_size, server_address, client_id)
        )
        thread.daemon = True  # Make thread daemon so it exits when main thread exits
        thread.start()
        
        images = []
        timeout_counter = 0
        max_timeout = 300
        
        while thread.is_alive() and timeout_counter < max_timeout:
            try:
                # Check for progress updates
                try:
                    progress = progress_queue.get_nowait()
                    current = progress['value']
                    total = progress['max']
                    percentage = int((current / total) * 100)
                    progress_bar.progress(current / total)
                    progress_text.text(f"Generating... ({percentage}%)")
                    timeout_counter = 0
                except queue.Empty:
                    pass
                
                # Check for new images
                try:
                    new_images = image_queue.get_nowait()
                    if new_images:
                        print(f"Debug: Received {len(new_images)} new images")
                        images.extend(new_images)
                        
                        # Create columns based on batch size
                        cols = st.columns(min(batch_size, 3))  # Max 3 images per row
                        
                        # Display images in a grid
                        for idx, img in enumerate(new_images):
                            if isinstance(img, Image.Image):
                                # Resize image to a more manageable size
                                width = 512  # You can adjust this value
                                ratio = width / img.size[0]
                                height = int(img.size[1] * ratio)
                                img_resized = img.resize((width, height), Image.Resampling.LANCZOS)
                                
                                # Display in the appropriate column
                                with cols[idx % 3]:
                                    st.image(
                                        img_resized, 
                                        use_column_width=True,
                                        caption=f"Generated Image {idx + 1}"
                                    )
                            else:
                                print(f"Debug: Unexpected image type: {type(img)}")
                        
                        if len(images) >= batch_size:
                            break
                        timeout_counter = 0
                except queue.Empty:
                    pass
                    
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Debug: Error in main loop: {str(e)}")
                break
        
        # Final progress update
        progress_bar.progress(1.0)
        progress_text.text("Generation complete! (100%)")
        
        return images
        
    except Exception as e:
        progress_text.text(f"Error: {str(e)}")
        progress_bar.progress(0)
        raise
        
    finally:
        # Clean up WebSocket connection
        if ws:
            try:
                ws.close()
            except:
                pass
        # Wait for thread to finish
        if 'thread' in locals() and thread.is_alive():
            thread.join(timeout=1.0)

def load_workflow():
    global _workflow_cache
    if _workflow_cache is not None:
        return _workflow_cache
        
    try:
        workflow_path = os.path.join(os.path.dirname(__file__), '..', 'workflows', 'lora-flux-api.json')
        with open(workflow_path, 'r') as f:
            _workflow_cache = json.load(f)
        print(f"Debug: Successfully loaded workflow from {workflow_path}")
        return _workflow_cache
    except Exception as e:
        print(f"Error loading workflow: {str(e)}")
        print(f"Attempted to load from: {workflow_path}")
        raise
