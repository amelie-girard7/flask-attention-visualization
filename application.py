import os
import json
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory, make_response
import pandas as pd
import torch
from bertviz import model_view
import matplotlib.pyplot as plt
import seaborn as sns

print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir(os.getcwd()))

# Configuration classes
class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///:memory:')
    DATA_PATH = os.getenv('DATA_PATH', 'data/model_2024-03-22-10/test_data_sample-attention.csv')
    ATTENTION_PATH = os.getenv('ATTENTION_PATH', 'data/model_2024-03-22-10/attentions')

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True

app = Flask(__name__)

# Use environment variable to determine which configuration to use
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

# Use configuration values
DATA_PATH = Path(app.config['DATA_PATH'])
ATTENTION_PATH = Path(app.config['ATTENTION_PATH'])

def clear_bertviz_cache():
    """
    Clear the cache files for BERTViz to prevent clutter and potential issues with old cache files.
    """
    try:
        bertviz_files = [f for f in os.listdir('/tmp') if f.startswith('attention_heatmap_')]
        for f in bertviz_files:
            os.remove(os.path.join('/tmp', f))
        print("BERTViz cache cleared successfully.")
    except Exception as e:
        print(f"Error clearing BERTViz cache: {str(e)}")

def load_data(file_path):
    """
    Load data from a CSV file.
    Returns a pandas DataFrame if the file exists, otherwise returns None.
    """
    if file_path is None or not file_path.exists():
        print(f"Data path {file_path} does not exist.")
        return None
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def get_attention_data(attention_path, story_id):
    """
    Load attention data for a given story ID.
    This function reads encoder, decoder, and cross-attention tensors and token data.
    Returns a tuple with attention data and tokens if successful, otherwise returns None.
    """
    attention_dir = attention_path / str(story_id)
    print(f"Loading attention data from {attention_dir}")

    if not attention_dir.exists():
        print(f"Attention directory does not exist: {attention_dir}")
        return None

    try:
        encoder_attentions = [torch.load(attention_dir / f'encoder_attentions_layer_{i}.pt') for i in range(12)]
        decoder_attentions = [torch.load(attention_dir / f'decoder_attentions_layer_{i}.pt') for i in range(12)]
        cross_attentions = [torch.load(attention_dir / f'cross_attentions_layer_{i}.pt') for i in range(12)]
    except Exception as e:
        print(f"Error loading attention tensors: {e}")
        return None

    try:
        with open(attention_dir / "tokens.json") as f:
            tokens = json.load(f)
    except Exception as e:
        print(f"Error loading tokens.json: {e}")
        return None

    encoder_text = tokens.get('encoder_text', [])
    generated_text = tokens.get('generated_text', "")
    generated_text_tokens = tokens.get('generated_text_tokens', [])

    print("Loaded encoder_text:", encoder_text)
    print("Loaded generated_text:", generated_text)
    print("Loaded generated_text_tokens:", generated_text_tokens)

    return encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens

@app.route('/')
def index():
    """
    Render the main index page of the application.
    """
    return render_template('index.html')

@app.route('/get_models', methods=['GET'])
def get_models():
    """
    Return a list of models available for selection.
    """
    print("get_models endpoint called")
    models = [{"key": "T5-base weight 1-1", "comment": "T5-base weight 1-1"}]
    return jsonify(models)

@app.route('/get_stories', methods=['POST'])
def get_stories():
    """
    Return a list of stories from the loaded data.
    """
    print("get_stories endpoint called")
    data = load_data(DATA_PATH)
    if data is None:
        print("Data not found")
        return jsonify({"error": "Data not found"}), 404
    print(f"Loaded data: {data.head()}")
    stories = data[['Premise', 'Initial', 'Original Ending', 'Counterfactual', 'Edited Ending', 'Generated Text']].to_dict(orient='records')
    return jsonify(stories)

@app.route('/fetch_story_data', methods=['POST'])
def fetch_story_data():
    """
    Return detailed information about a specific story given its index.
    """
    story_index = request.json.get('story_index')
    if story_index is None:
        return jsonify({"error": "Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data = load_data(DATA_PATH)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    story = data.iloc[story_index].to_dict()
    return jsonify(story)

@app.route('/visualize_attention', methods=['POST'])
def visualize_attention():
    """
    Generate and return the attention heatmap for a specific story.
    """
    story_index = request.json.get('story_index')
    if story_index is None:
        return jsonify({"error": "Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data = load_data(DATA_PATH)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    story_id = data.iloc[story_index]["StoryID"]

    try:
        result = get_attention_data(ATTENTION_PATH, story_id)
        if result is None:
            return jsonify({"error": "Error loading attention data"}), 500
        encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens = result
        print(f"Attention data loaded for story index {story_index}")
        print(f"Generated Text Tokens: {generated_text_tokens}")
    except Exception as e:
        print(f"Error loading attention data: {str(e)}")
        return jsonify({"error": str(e)}), 500

    try:
        first_layer_attention = cross_attentions[0]
        if isinstance(first_layer_attention, tuple):
            first_layer_attention = first_layer_attention[0]
        first_batch_attention = first_layer_attention[0]
        print("Shape of first batch attention:", first_batch_attention.shape)

        if first_batch_attention.ndim == 3:
            attention_to_plot = first_batch_attention.mean(axis=0)
            print("Averaged attention shape:", attention_to_plot.shape)
        elif first_batch_attention.ndim == 2:
            attention_to_plot = first_batch_attention
        else:
            print(f"Unexpected attention matrix dimension: {first_batch_attention.ndim}D")
            raise ValueError(f"Unexpected attention matrix dimension: {first_batch_attention.ndim}D")

        image_path = f'/tmp/attention_heatmap_{story_id}.png'
        plot_attention_heatmap(attention_to_plot, encoder_text, generated_text_tokens, "Cross-Attention Weights (First Layer)", image_path)
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"image_path": image_path})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve the generated heatmap image.
    """
    return send_from_directory('/tmp', filename)

@app.route('/visualize_model_view', methods=['POST'])
def handle_visualize_model_view():
    """
    Handle the request to visualize the model view of the attention mechanism.
    """
    return visualize_model_view(request, lambda _: load_data(DATA_PATH), get_attention_data, ATTENTION_PATH)

def plot_attention_heatmap(attention, x_tokens, y_tokens, title, image_path):
    """
    Plot and save an attention heatmap.
    
    Parameters:
    attention (numpy.ndarray): The attention weights to be visualized.
    x_tokens (list of str): The input tokens.
    y_tokens (list of str): The generated text tokens.
    title (str): The title for the heatmap.
    image_path (str): The path to save the generated heatmap image.
    """
    print(f"Number of x_tokens (input): {len(x_tokens)}")
    print(f"Number of y_tokens (generated text): {len(y_tokens)}")
    print(f"Attention matrix shape: {attention.shape}")

    if attention.shape[-1] != len(x_tokens) or attention.shape[-2] != len(y_tokens):
        print("Attention dimensions do not match the token list dimensions.")
        return

    fig_width = max(15, len(x_tokens) / 2)
    fig_height = max(10, len(y_tokens) / 2)

    plt.figure(figsize=(fig_width, fig_height))
    print("Attention matrix shape for plotting:", attention.shape)
    print("Number of input tokens:", len(x_tokens))
    print("Number of output tokens:", len(y_tokens))

    sns.heatmap(attention, xticklabels=x_tokens, yticklabels=y_tokens, cmap='viridis', cbar=True)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel('Input Tokens', fontsize=12)
    plt.ylabel('Generated Text Tokens', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    plt.savefig(image_path)
    plt.close()

def visualize_model_view(request, load_data, get_attention_data, ATTENTION_PATH):
    """
    Generate and return the model view visualization for the attention mechanism.
    
    Parameters:
    request (Request): The Flask request object.
    load_data (function): Function to load story data.
    get_attention_data (function): Function to load attention data.
    ATTENTION_PATH (Path): Path to the attention data directory.
    
    Returns:
    Response: The generated HTML content for the model view visualization.
    """
    story_index = request.json.get('story_index')
    if story_index is None:
        return jsonify({"error": "Story index not provided"}), 400

    try:
        story_index = int(story_index)
    except ValueError:
        return jsonify({"error": "Invalid story index"}), 400

    data = load_data(DATA_PATH)
    if data is None:
        return jsonify({"error": "Data not found"}), 404

    story_id = data.iloc[story_index]["StoryID"]

    try:
        result = get_attention_data(ATTENTION_PATH, story_id)
        if result is None:
            return jsonify({"error": "Error loading attention data"}), 500
        encoder_attentions, decoder_attentions, cross_attentions, encoder_text, generated_text, generated_text_tokens = result
        print(f"Attention data loaded for story index {story_index}")
        print(f"Generated Text Tokens: {generated_text_tokens}")
    except Exception as e:
        print(f"Error loading attention data: {str(e)}")
        return jsonify({"error": str(e)}), 500

    try:
        html_content = model_view(
            encoder_attention=encoder_attentions,
            decoder_attention=decoder_attentions,
            cross_attention=cross_attentions,
            encoder_tokens=encoder_text,
            decoder_tokens=generated_text_tokens,
            html_action='return'
        )
        print("HTML content generated successfully")
        response = make_response(html_content.data)
        response.headers['Content-Type'] = 'text/html'
    except Exception as e:
        print(f"Error generating model view: {str(e)}")
        return jsonify({"error": str(e)}), 500

    return response

if __name__ == '__main__':
    clear_bertviz_cache()  # Clear BERTViz cache at the beginning
    app.run(debug=app.config['DEBUG'])
