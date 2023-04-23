import threading
from flask import Flask, request, jsonify
from time import sleep
import os
from zotero import run_zotero, validate_credentials

app = Flask(__name__)

@app.route("/zotero", methods=["GET"])
def zotero():
    if request.method == "GET":
        user_id = request.args.get("user_id")
        api_key = request.args.get("api_key")

        if user_id is None or api_key is None:
            return jsonify({"error": "user_id and api_key are required"}), 400

        if not validate_credentials(user_id, api_key):
            return jsonify({"error": "Invalid user_id or api_key"}), 401

        def run_in_background():
            sleep(1)
            run_zotero(user_id, api_key)

        threading.Thread(target=run_in_background).start()
        return jsonify({"message": "Zotero process started"}), 202

    return jsonify({"error": "Invalid request method"}), 405



