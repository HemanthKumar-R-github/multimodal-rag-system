"""Flask blueprints: server-rendered pages + JSON API."""
import json

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .evaluation import CONFIGS, Evaluator, load_dataset

pages = Blueprint("pages", __name__)
api = Blueprint("api", __name__, url_prefix="/api")


def pipeline():
    return current_app.extensions["docintel"]


# Pages -------------------------------------------------------------------
@pages.get("/")
def ask_page():
    return render_template("ask.html", active="ask")


@pages.get("/documents")
def documents_page():
    return render_template("documents.html", active="documents")


@pages.get("/evaluate")
def evaluate_page():
    return render_template("evaluate.html", active="evaluate", configs=list(CONFIGS))


# API ---------------------------------------------------------------------
@api.get("/status")
def status():
    return jsonify(pipeline().status())


@api.get("/documents")
def list_documents():
    return jsonify(pipeline().store.list_documents())


@api.post("/documents")
def upload_documents():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files provided under field 'files'."), 400
    created, errors = [], []
    for f in files:
        try:
            created.append(pipeline().ingest_upload(f))
        except ValueError as exc:
            errors.append(str(exc))
    return jsonify(documents=created, errors=errors), (201 if created else 400)


@api.get("/documents/<int:doc_id>/chunks")
def document_chunks(doc_id: int):
    doc = pipeline().store.get_document(doc_id)
    if not doc:
        abort(404)
    return jsonify(document=doc, chunks=pipeline().store.chunks_for_document(doc_id))


@api.delete("/documents/<int:doc_id>")
def delete_document(doc_id: int):
    if not pipeline().delete_document(doc_id):
        abort(404)
    return jsonify(ok=True)


@api.post("/query")
def query():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return jsonify(error="question is required"), 400
    if len(question) > 2000:
        return jsonify(error="question is too long"), 400
    mode = body.get("mode", "hybrid")
    if mode not in {"bm25", "dense", "hybrid"}:
        return jsonify(error="mode must be bm25, dense or hybrid"), 400
    top_k = max(1, min(int(body.get("top_k", 5)), 15))
    result = pipeline().answer(question, mode=mode, rerank=bool(body.get("rerank", True)),
                               top_k=top_k, use_llm=bool(body.get("use_llm", True)))
    return jsonify(result)


@api.get("/assets/<path:filename>")
def asset(filename: str):
    return send_from_directory(current_app.config["ASSET_DIR"], filename)


# Evaluation --------------------------------------------------------------
@api.get("/eval/datasets")
def eval_datasets():
    eval_dir = current_app.config["EVAL_DIR"]
    out = []
    for path in sorted(eval_dir.glob("*.json")):
        try:
            data = load_dataset(path)
            out.append({"file": path.name, "name": data.get("name", path.stem), "n": len(data.get("items", []))})
        except (OSError, json.JSONDecodeError):
            continue
    return jsonify(out)


@api.post("/eval/synthesize")
def eval_synthesize():
    body = request.get_json(silent=True) or {}
    n = max(1, min(int(body.get("n", 10)), 50))
    data = Evaluator(pipeline()).synthesize(n)
    if not data["items"]:
        return jsonify(error="Index some documents first."), 400
    path = current_app.config["EVAL_DIR"] / f"{data['name']}.json"
    path.write_text(json.dumps(data, indent=2))
    return jsonify(file=path.name, **data)


@api.post("/eval/run")
def eval_run():
    body = request.get_json(silent=True) or {}
    if "dataset" in body and isinstance(body["dataset"], dict):
        dataset, name = body["dataset"], body["dataset"].get("name", "inline")
    else:
        fname = secure_filename(str(body.get("file", "")))
        path = current_app.config["EVAL_DIR"] / fname
        if not fname or not path.exists():
            return jsonify(error="Unknown dataset file"), 400
        dataset, name = load_dataset(path), fname
    if not dataset.get("items"):
        return jsonify(error="Dataset has no items"), 400
    options = {
        "configs": body.get("configs") or list(CONFIGS),
        "k": max(1, min(int(body.get("k", 5)), 20)),
        "answer_config": body.get("answer_config", "hybrid+rerank"),
        "generate": bool(body.get("generate", True)),
        "use_llm": bool(body.get("use_llm", True)),
        "llm_judge": bool(body.get("llm_judge", False)),
    }
    result = Evaluator(pipeline()).run(dataset, **options)
    config = {**options, "backends": pipeline().status()["backends"]}
    run_id = pipeline().store.add_eval_run(name, config, result["summary"], result["details"])
    return jsonify(id=run_id, dataset=name, config=config, **result)


@api.get("/eval/runs")
def eval_runs():
    return jsonify(pipeline().store.list_eval_runs())


@api.get("/eval/runs/<int:run_id>")
def eval_run_detail(run_id: int):
    run = pipeline().store.get_eval_run(run_id)
    if not run:
        abort(404)
    return jsonify(run)
