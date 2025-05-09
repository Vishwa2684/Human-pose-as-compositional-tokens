import gdown
import os

# Replace this with your public Google Drive shareable link

def load_decoder_and_quantizer_weights():
    if(os.path.exists("./weights/50.pt")):
        print("Weights already downloaded")
        return
    # Extract the file ID from the shareable link (e.g., "FILE_ID" in the link
    url = "https://drive.google.com/file/d/1an2hhD202GZdOmz7cRYSsc4LDJFNMvMK/view?usp=sharing"
    output_dir = "./weights"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "50.pt")  # Replace with desired filename

    # Download the file
    gdown.download(url, output_path, quiet=False)
