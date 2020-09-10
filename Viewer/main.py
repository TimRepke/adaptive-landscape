from argparse import ArgumentParser
from flask import Flask, Blueprint, jsonify
import json
import os

parser = ArgumentParser(description='Create graph from parsed corpus.')
parser.add_argument('--data', dest='data', type=str, help='Directory containing reduced *.edges files')
args = parser.parse_args()

app = Flask(__name__,
            static_folder='./webroot/',
            static_url_path='/')

data_server = Blueprint('data_server', __name__, static_folder=args.data, static_url_path='/static')


@data_server.route('/list')
def list_datasets():
    return jsonify([fn for fn in os.listdir(args.data)])


app.register_blueprint(data_server, url_prefix='/data')
app.run()
