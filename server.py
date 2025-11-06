from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import re
import json

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files['image']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
        image_file.save(temp_img.name)
        temp_path = temp_img.name

    try:
        result = subprocess.run(
            ["python3", "app.py", temp_path],
            capture_output=True,
            text=True,
            check=True
        )

        raw_output = (result.stdout + result.stderr).strip()

        # Clean terminal formatting and logs
        cleaned_output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw_output)
        cleaned_output = re.sub(r'[^\x00-\x7F]+', '', cleaned_output)
        cleaned_output = re.sub(r'Matching:\s*\d+%.*', '', cleaned_output)

        # Extract JSON block
        json_match = re.search(r'\{[\s\S]*\}$', cleaned_output)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed_json = json.loads(json_str)
                # Extract only needed fields
                filtered_result = {
                    "accuracy": parsed_json.get("accuracy"),
                    "imagename": parsed_json.get("imagename"),
                    "max_score": parsed_json.get("max_score"),
                    "consistency": parsed_json.get("consistency"),
                    "objects_detected": parsed_json.get("objects_detected")
                }
                return jsonify(filtered_result)
            except json.JSONDecodeError:
                pass

        # Fallback: return cleaned text
        flat_output = cleaned_output.replace("\n", " ").replace("\r", " ").strip()
        return jsonify({"result": flat_output})

    except subprocess.CalledProcessError as e:
        error_output = (e.stderr or e.stdout or "Unknown error").strip()
        error_output = error_output.replace("\n", " ").replace("\r", " ")
        return jsonify({"error": error_output}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
