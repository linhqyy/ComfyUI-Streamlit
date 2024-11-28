import requests
import json
import urllib.request
import urllib.parse
import time
import os
import random
import copy
import io

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import websocket
from io import BytesIO


@st.cache_data  
def image_thumbnail(array, max_width=800, max_height=600):
    image = Image.fromarray(array.astype('uint8'))
    # 이미지의 크기를 조절
    image.thumbnail((max_width, max_height), Image.LANCZOS)
    return image


def update_workflow(workflow, prompt, seed, batch_size, lora_model="None", lora_strength=0.8):
    try:
        workflow_copy = copy.deepcopy(workflow)
        
        # Update CLIP Text Encode node (node 5)
        if "5" in workflow_copy:
            workflow_copy["5"]["inputs"]["text"] = prompt

        # Update KSampler node for seed (node 7)
        if "7" in workflow_copy:
            if seed == -1:
                seed = random.randint(0, 2**32-1)
            workflow_copy["7"]["inputs"]["noise_seed"] = seed

        # Update Empty Latent Image node for batch size (node 6)
        if "6" in workflow_copy:
            workflow_copy["6"]["inputs"]["batch_size"] = int(batch_size)

        # Update LoRA node (node 3)
        if "3" in workflow_copy:
            if lora_model != "None":
                workflow_copy["3"]["inputs"]["lora_01"] = f"{lora_model}.safetensors"
                workflow_copy["3"]["inputs"]["strength_01"] = float(lora_strength)
            else:
                workflow_copy["3"]["inputs"]["lora_01"] = "None"
                workflow_copy["3"]["inputs"]["strength_01"] = 0.0

        print(f"Debug: Updated workflow parameters:")
        print(f"  - Prompt: {prompt}")
        print(f"  - Seed: {seed}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - LoRA: {lora_model} (strength: {lora_strength})")
        
        return workflow_copy
        
    except Exception as e:
        print(f"Debug: Error updating workflow: {str(e)}")
        print(f"Debug: Original workflow structure: {json.dumps(workflow, indent=2)}")
        raise


def queue_workflow(server_address, client_id, prompt, batch_size, seed, workflow):
    """Queue a workflow to the ComfyUI server"""
    try:
        # Get the selected LoRA and strength from session state if available
        lora_model = st.session_state.get('selected_lora', 'None')
        lora_strength = st.session_state.get('lora_strength', 1.0)
        
        # Update the workflow with all parameters
        updated_workflow = update_workflow(
            workflow=workflow,
            prompt=prompt,
            seed=seed,
            batch_size=batch_size,
            lora_model=lora_model,
            lora_strength=lora_strength
        )
        
        # Format the request data properly
        prompt_workflow = {
            "prompt": updated_workflow,
            "client_id": client_id
        }
        
        print("Debug: Queueing workflow to", server_address)
        print("Debug: Sending workflow data:", json.dumps(prompt_workflow, indent=2))
        
        response = requests.post(f"http://{server_address}/prompt", json=prompt_workflow)
        
        if response.status_code != 200:
            print(f"Debug: Server response: {response.text}")
            raise Exception(f"Failed to queue workflow: {response.text}")
            
        result = response.json()
        return result["prompt_id"]
        
    except Exception as e:
        print(f"Error queueing workflow: {str(e)}")
        raise

def get_queue(server_address):
    # create the GET request
    req = urllib.request.Request(f"http://{server_address}/queue", method='GET')

    # sending the request and getting the response
    with urllib.request.urlopen(req) as response:
        response_data = json.loads(response.read().decode('utf-8'))

        return response_data

def cancel_running(server_address):
    url = f"http://{server_address}/interrupt"
    req_headers = {'Content-Type': 'application/json'}    
    interrupt_request = urllib.request.Request(url, headers=req_headers, method='POST')

    # send request and get the response
    with urllib.request.urlopen(interrupt_request) as response:
        return response
    
def delete_queue(server_address, client_id):
    response = get_queue(server_address)
    try:
        task = response["queue_running"][0]
        if task[-2]["client_id"] == client_id:
            cancel_running(server_address)

    except:
        pass
        # st.toast("No queue")
   

def receive_images(ws, prompt_id, image_queue, progress_queue, batch_size, server_address, client_id):
    try:
        while True:
            try:
                message = ws.recv()
                if not message:
                    continue
                    
                try:
                    message = json.loads(message)
                except json.JSONDecodeError:
                    continue
                
                if message.get("type") == "progress":
                    progress_queue.put(message["data"])
                    
                elif message.get("type") == "executed":
                    if message.get("data", {}).get("node") is None:
                        continue
                    # Handle image data
                    if "images" in message.get("data", {}).get("output", {}):
                        images = message["data"]["output"]["images"]
                        print(f"Debug: Processing {len(images)} images from server")
                        
                        # Get the images from the server
                        processed_images = []
                        for image_data in images:
                            try:
                                # Parse the image data dictionary
                                if isinstance(image_data, dict):
                                    filename = image_data.get('filename', '')
                                    subfolder = image_data.get('subfolder', '')
                                    type_name = image_data.get('type', 'temp')
                                    
                                    # Construct the proper URL
                                    image_url = f"http://{server_address}/api/view?filename={filename}&subfolder={subfolder}&type={type_name}"
                                    print(f"Debug: Fetching image from {image_url}")
                                    
                                    response = requests.get(image_url)
                                    if response.status_code == 200:
                                        img = Image.open(BytesIO(response.content))
                                        processed_images.append(img)
                                    else:
                                        print(f"Debug: Failed to fetch image. Status code: {response.status_code}")
                                        print(f"Debug: Response content: {response.text}")
                            except Exception as e:
                                print(f"Debug: Error processing image data: {str(e)}")
                                print(f"Debug: Image data: {image_data}")
                                continue
                        
                        if processed_images:
                            image_queue.put(processed_images)
                            print(f"Debug: Queued {len(processed_images)} processed images")
                        
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                print(f"Debug: Error in receive loop: {str(e)}")
                break
                
    except Exception as e:
        print(f"Debug: Error in receive_images: {str(e)}")
    finally:
        try:
            ws.close()
        except:
            pass


def get_available_loras(comfyui_path="/workspace/ComfyUI"):
    """Get all available LoRA models from the ComfyUI/models/loras directory"""
    lora_path = os.path.join(comfyui_path, "models", "loras")
    
    # Check if directory exists
    if not os.path.exists(lora_path):
        print(f"Debug: LoRA directory not found at {lora_path}")
        return ["None"]
    
    # Get all .safetensors files
    lora_files = []
    for file in os.listdir(lora_path):
        if file.endswith(('.safetensors', '.ckpt', '.pt')):
            # Remove file extension for display
            name = os.path.splitext(file)[0]
            lora_files.append(name)
    
    # Sort alphabetically and ensure "None" is first
    lora_files.sort()
    if "None" not in lora_files:
        lora_files.insert(0, "None")
    
    print(f"Debug: Found {len(lora_files)} LoRA models")
    return lora_files

