from flask import Flask, request, jsonify, send_from_directory
from diffusers import StableDiffusionPipeline
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"

# Load the Stable Diffusion model
print("Loading model...")
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to("cpu") 
print("Model loaded successfully!")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')  

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        # Get the prompt from the request
        data = request.get_json()
        prompt = data.get("prompt", "")
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Generate the image
        print(f"Generating image for prompt: {prompt}")
        result = pipeline(prompt)
        
        if not result.images:
            raise ValueError("No images were generated.")
        
        image = result.images[0]

        # Check the image size and log it
        width, height = image.size
        print(f"Generated image dimensions: {width}x{height}")
        
        if width <= 0 or height <= 0:
            return jsonify({"error": "Generated image has invalid dimensions."}), 500

        # Save the image to the static folder
        filename = "generated_image.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(filepath)

        # Return the file path to the frontend
        return jsonify({"image_url": f"/static/{filename}"})
    
    except Exception as e:
        print(f"Error during image generation: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Serve static files
@app.route("/static/<path:filename>")
def serve_static_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
