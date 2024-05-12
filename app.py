from flask import Flask, request, jsonify
from transformer import LowercaseTransformer
import pickle

app = Flask(__name__)

with open('simple_text_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    message = request.data.decode('utf-8')
    print(message)
    inference = model.predict([message])[0]
    return inference


if __name__ == '__main__':
    app.run()
