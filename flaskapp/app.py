from flask import Flask, jsonify, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

app = Flask(__name__)

# Paths to the CSV data files
DATA_PATHS = {
    "T5-base weight 1-1": Path('data/model_2024-03-22-10/test_details_sample.csv'),
    # Add other models if needed
}

# Function to load data from the CSV file
def load_data(model_key):
    data_path = DATA_PATHS.get(model_key)
    if data_path is None or not data_path.exists():
        return None
    return pd.read_csv(data_path)

# Function to get attention data
def get_attention_data(data, index):
    # Create the input sequence
    input_sequence = (
        f"{data['Premise'].iloc[index]}"
        f"{data['Initial'].iloc[index]}"
        f"{data['Original Ending'].iloc[index]} </s> "
        f"{data['Premise'].iloc[index]} {data['Counterfactual'].iloc[index]}"
    )

    # Dummy attention data for demonstration purposes
    # Replace with your actual attention extraction logic
    attention_data = {
        'encoder_text': input_sequence.split(),
        'generated_text': data['Generated Text'].iloc[index].split(),
        'attention': [[0.1] * len(input_sequence.split())] * len(data['Generated Text'].iloc[index].split())
    }
    return attention_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    models = [{"key": key, "comment": key} for key in DATA_PATHS.keys()]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    model_key = request.json.get('model_key')
    data = load_data(model_key)
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/visualize_attention', methods=['POST'])
def visualize_attention():
    model_key = request.json.get('model_key')
    story_index = request.json.get('story_index')
    if story_index is None:
        return jsonify({"error": "Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data = load_data(model_key)
    if data is None:
        return jsonify({"error": "Data not found"}), 404
    
    attention_data = get_attention_data(data, story_index)
    encoder_text = attention_data['encoder_text']
    generated_text = attention_data['generated_text']
    attention = attention_data['attention']

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=encoder_text, yticklabels=generated_text, cmap='viridis', cbar=True)
    plt.xticks(rotation=90)
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title('Attention Heatmap')

    image_path = '/tmp/attention_heatmap.png'
    plt.savefig(image_path)
    plt.close()

    return send_file(image_path, mimetype='image/png')

@app.route('/visualize_model_view', methods=['POST'])
def visualize_model_view():
    # Here you should implement logic to create the model view
    # This is just a placeholder
    html_content = "<html><body><h1>Model View</h1></body></html>"
    html_path = '/tmp/model_view.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    return send_file(html_path, mimetype='text/html')

@app.route('/visualize_head_view', methods=['POST'])
def visualize_head_view():
    # Here you should implement logic to create the head view
    # This is just a placeholder
    html_content = "<html><body><h1>Head View</h1></body></html>"
    html_path = '/tmp/head_view.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    return send_file(html_path, mimetype='text/html')

@app.route('/model_view')
def serve_model_view():
    return send_file('/tmp/model_view.html', mimetype='text/html')

@app.route('/head_view')
def serve_head_view():
    return send_file('/tmp/head_view.html', mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
