"""Flask entry point: `python app.py` or `flask --app app run`."""
import logging
import os

from flask import Flask, jsonify

from config import Config
from docintel.pipeline import Pipeline
from docintel.web import api, pages


def create_app(config_class=Config) -> Flask:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.extensions["docintel"] = Pipeline(config_class)
    app.register_blueprint(pages)
    app.register_blueprint(api)

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return resp

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File too large"), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Not found"), 404

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
