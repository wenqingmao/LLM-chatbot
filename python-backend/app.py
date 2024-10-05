from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from langgraph_bot import get_final_answer

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

@app.route('/chat', methods=['POST'])
def chat():
    # Get the user input (question) from the request
    data = request.get_json()
    user_question = data.get('message')

    # Get the final answer from the chatbot
    answer = get_final_answer(user_question)

    # Return the answer as JSON
    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

