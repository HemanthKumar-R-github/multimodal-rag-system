import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docintel.evaluation import metrics as M  # noqa: E402
from docintel.generation import extract_citations  # noqa: E402
from docintel.ingestion.chunker import chunk_text, table_to_markdown  # noqa: E402
from docintel.retrieval.bm25_index import BM25Index  # noqa: E402
from docintel.retrieval.hybrid import reciprocal_rank_fusion  # noqa: E402


def test_rrf_rewards_agreement():
    fused = dict(reciprocal_rank_fusion({"bm25": [1, 2, 3], "dense": [3, 1, 4]}, k=60))
    assert max(fused, key=fused.get) == 1  # ranked 1st and 2nd
    assert fused[3] > fused[2] and fused[3] > fused[4]
    assert abs(fused[1] - (1 / 61 + 1 / 62)) < 1e-12


def test_bm25_ranks_exact_term_first():
    idx = BM25Index()
    idx.build([
        {"id": 1, "content": "battery replacement after 3,000 cycles", "filename": "a"},
        {"id": 2, "content": "emergency stop button halts the robot", "filename": "b"},
    ])
    assert idx.search("emergency stop")[0][0] == 2


def test_chunker_overlap_and_table():
    text = " ".join(f"word{i}." for i in range(400))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) >= 4 and all(len(c.split()) <= 100 for c in chunks)
    md = table_to_markdown([["A", "B"], ["1", None]])
    assert md.splitlines()[0] == "| A | B |" and md.splitlines()[2] == "| 1 |  |"


def test_retrieval_metrics():
    retrieved = [{"filename": "x.pdf", "page": 2, "content": "no"}, {"filename": "y.pdf", "page": 1, "content": "yes"}]
    m = M.retrieval_metrics(retrieved, [{"doc": "y.pdf"}], k=2)
    assert m["hit@2"] == 1.0 and m["mrr"] == 0.5 and m["recall@2"] == 1.0 and m["precision@2"] == 0.5


def test_answer_metrics_and_citations():
    assert extract_citations("Revenue rose [1][3] and [2, 9].", 3) == [1, 3, 2]
    assert M.token_f1("revenue was 48.2 million", "revenue 48.2 million") > 0.8
    assert M.rouge_l("quarterly revenue grew strongly", "quarterly revenue grew") > 0.8
    assert M.faithfulness("Revenue was $48.2 million [1].", ["Total revenue reached $48.2 million"]) == 1.0
    assert M.citation_coverage("Fact one [1]. Fact two.") == 0.5
