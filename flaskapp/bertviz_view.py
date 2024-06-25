from flask import jsonify, make_response
from bertviz import model_view

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

    data = load_data()
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
        print(f"HTML content generated successfully")
        response = make_response(html_content.data)
        response.headers['Content-Type'] = 'text/html'
    except Exception as e:
        print(f"Error generating model view: {str(e)}")
        return jsonify({"error": str(e)}), 500

    return response
