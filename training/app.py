from flask import Flask, session, jsonify, request, make_response
import os
from inference import generate_caption

app = Flask(__name__)


@app.route('/captionify', methods=['GET', 'POST'])
def captionify():

    data = request.json
    img_path = data['path']
    return make_response(generate_caption(img_path))


if __name__ == '__main__':
    app.run(debug=False)

