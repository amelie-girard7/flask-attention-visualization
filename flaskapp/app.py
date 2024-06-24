from flask import Flask, jsonify, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

app = Flask(__name__)

DATA_PATH = Path('data/test_details_sample.csv')

def load_data():
    return pd.read_csv(DATA_PATH)

def get_attention_data():
    # Dummy attention data for demonstration purposes
    # Replace with your actual attention extraction logic
    data = load_data()
    attention_data = {
        'encoder_text': data['Premise'].iloc[0].split(),
        'generated_text': data['Generated Text'].iloc[0].split(),
        'attention': [[0.1] * len(data['Premise'].iloc[0].split())] * len(data['Generated Text'].iloc[0].split())
    }
    return attention_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attention', methods=['POST'])
def visualize_attention():
    attention_data = get_attention_data()
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

if __name__ == '__main__':
    app.run(debug=True)
