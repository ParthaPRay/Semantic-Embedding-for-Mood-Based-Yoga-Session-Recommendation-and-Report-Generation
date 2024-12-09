import gradio as gr
import requests
import time
import os
import socket

# Configuration Variables
FASTAPI_HOST = "localhost"  # FastAPI is running on the same Raspberry Pi
FASTAPI_PORT = 5000
GRADIO_PORT = 7860

FASTAPI_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/process_prompt"
DOWNLOAD_ENDPOINT_TEMPLATE = "http://{pi_ip}:{fastapi_port}/download_report/{filename}"

def get_local_ip():
    """
    Automatically retrieve the Raspberry Pi's local IP address.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a non-routable address to get the local IP
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Retrieve Pi's local IP
pi_local_ip = get_local_ip()

# Ensure FastAPI is accessible from Gradio
# If FastAPI is bound to 0.0.0.0, 'localhost' is fine here since Gradio is on the same machine

def process_mood_prompt(prompt):
    """
    Send the user's mood prompt to the FastAPI backend and handle the response.
    """
    if not prompt.strip():
        return "Please enter a valid mood status.", None

    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}

    try:
        # Send POST request to FastAPI backend
        response = requests.post(FASTAPI_URL, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            
            if status == "no_match":
                final_comment = data.get("message", "No suitable Yoga Asana found for your current mood.")
                return final_comment, None
            elif status == "success":
                final_comment = data.get("final_comment", "No comment available.")
                report_url = data.get("download_report_url")
                if report_url:
                    # Extract the filename from the download URL
                    filename = os.path.basename(report_url)
                    # Construct the full download URL using the Pi's IP
                    full_report_url = DOWNLOAD_ENDPOINT_TEMPLATE.format(
                        pi_ip=pi_local_ip,
                        fastapi_port=FASTAPI_PORT,
                        filename=filename
                    )
                    return final_comment, full_report_url
                else:
                    return final_comment, None
            else:
                return "Unexpected response status.", None
        else:
            return f"Error: {response.status_code} - {response.text}", None

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again.", None
    except requests.exceptions.ConnectionError:
        return "Failed to connect to the backend server. Ensure FastAPI is running.", None
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", None

# Define Gradio Interface
with gr.Blocks(css="""
    .gradio-container {
        max-width: 700px;
        margin: auto;
    }
    .title {
        text-align: center;
        color: #4A90E2;
    }
    .submit-button {
        background-color: #4A90E2;
        color: white;
    }
    .output-text {
        background-color: #F0F4F8;
        padding: 15px;
        border-radius: 5px;
    }
""") as demo:
    # Title and Description
    gr.Markdown("<h1 class='title'>🧘‍♂️ Mood-Based Yoga Asana Recommendation</h1>")
    gr.Markdown("### Tell us about your current mood status, and we'll recommend a Yoga Asana for you!")

    # Input Section
    with gr.Row():
        with gr.Column():
            mood_input = gr.Textbox(
                label="Your Mood Status",
                placeholder="e.g., I feel overwhelmed and stressed with my work.",
                lines=4
            )
            submit_btn = gr.Button("Submit", variant="primary", elem_id="submit-button")
    
    # Output Section
    with gr.Row():
        with gr.Column():
            final_comment = gr.Textbox(
                label="Final Comment",
                lines=6,
                interactive=False,
                elem_classes="output-text"
            )
            report_link = gr.Markdown("")
    
    # Define the processing function
    def on_submit(prompt):
        if not prompt.strip():
            return "Please enter a valid mood status.", None
        
        # Start processing (Gradio shows a loading indicator automatically)
        final_comment_text, report_url = process_mood_prompt(prompt)
        
        # Prepare outputs
        if report_url:
            report_md = f"[Download Your PDF Report]({report_url})"
        else:
            report_md = ""
        
        return final_comment_text, report_md

    # Connect the Submit button to the processing function
    submit_btn.click(
        fn=on_submit,
        inputs=[mood_input],
        outputs=[final_comment, report_link],
        show_progress=True  # This shows Gradio's built-in loading indicator
    )

# Launch the Gradio app
demo.launch(
    server_name="0.0.0.0",  # Bind to all network interfaces to allow external access
    server_port=GRADIO_PORT,  # Specify the port (default is 7860)
    share=False,              # Disable public sharing via Gradio
    debug=True                # Enable debug mode for verbose logs (optional)
)
