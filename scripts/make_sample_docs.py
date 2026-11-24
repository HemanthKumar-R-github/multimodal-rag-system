"""Generate a small multimodal demo corpus (native text, ruled table, raster chart,
scanned page) and optionally ingest it.

    python scripts/make_sample_docs.py            # write files to data/samples/
    python scripts/make_sample_docs.py --ingest   # also index them
"""
import argparse
import sys
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "samples"


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def bar_chart(path: Path) -> None:
    data = [("North America", 18.4), ("Europe", 12.1), ("APAC", 13.9), ("LATAM", 3.8)]
    img = Image.new("RGB", (900, 560), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Revenue by region, FY2025 ($M)", fill="#111", font=_font(28))
    base, left, width, gap, scale = 480, 90, 140, 50, 20
    d.line((left - 20, base, 860, base), fill="#444", width=2)
    for i, (label, value) in enumerate(data):
        x = left + i * (width + gap)
        d.rectangle((x, base - value * scale, x + width, base), fill="#0f766e")
        d.text((x + 30, base - value * scale - 34), f"{value}", fill="#111", font=_font(24))
        d.text((x, base + 14), label, fill="#333", font=_font(20))
    img.save(path)


def scanned_page(path: Path) -> None:
    img = Image.new("RGB", (1240, 1754), "#fbfaf6")
    d = ImageDraw.Draw(img)
    lines = [
        "MEMO - Board of Directors (scanned copy)",
        "",
        "Date: 14 January 2026",
        "Subject: Approval of the 2026 capital plan",
        "",
        "The board approved a capital expenditure budget of $6.5 million",
        "for 2026, of which $2.2 million is allocated to the new Singapore",
        "data centre. The dividend was held at $0.42 per share.",
        "",
        "Signed, M. Okafor, Company Secretary",
    ]
    y = 140
    for ln in lines:
        d.text((120, y), ln, fill="#222", font=_font(34))
        y += 56
    img = img.rotate(0.6, fillcolor="#fbfaf6")
    img.save(path)


def annual_report(path: Path, chart: Path, scan: Path) -> None:
    doc = fitz.open()

    p = doc.new_page()
    p.insert_text((72, 80), "Acme Robotics - Annual Report 2025", fontsize=20)
    body = (
        "Executive summary. Acme Robotics delivered record results in fiscal year 2025. Total revenue "
        "reached $48.2 million, an increase of 23% over 2024. Growth was driven primarily by the APAC "
        "region, where revenue grew 41% following the launch of the R7 warehouse robot in Singapore and "
        "Japan.\n\nOperating margin improved from 17% to 20% thanks to lower component costs and a shift "
        "toward software subscriptions, which now represent 31% of revenue. Net income was $7.9 million.\n\n"
        "Headcount grew to 412 employees at year end. The company opened a new engineering hub in Kraków, "
        "Poland, focused on perception and motion-planning software."
    )
    p.insert_textbox(fitz.Rect(72, 110, 540, 760), body, fontsize=11)

    p = doc.new_page()
    p.insert_text((72, 80), "Quarterly performance", fontsize=16)
    p.insert_textbox(fitz.Rect(72, 96, 540, 150),
                     "Table 1: Quarterly financial results for fiscal year 2025.", fontsize=10)
    rows = [["Quarter", "Revenue ($M)", "Operating margin", "Customers"],
            ["Q1", "10.1", "18%", "1,240"], ["Q2", "11.4", "19%", "1,390"],
            ["Q3", "12.6", "21%", "1,560"], ["Q4", "14.1", "22%", "1,720"]]
    x0, y0, cw, rh = 72, 150, 117, 26
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p.draw_rect(rect, color=(0.3, 0.3, 0.3), width=0.8)
            p.insert_text((rect.x0 + 6, rect.y0 + 17), cell, fontsize=10)
    p.insert_textbox(fitz.Rect(72, 300, 540, 760),
                     "Q4 was the strongest quarter, with revenue of $14.1 million and an operating margin of 22%. "
                     "Customer count rose every quarter, ending the year at 1,720 active customers.", fontsize=11)

    p = doc.new_page()
    p.insert_text((72, 80), "Regional breakdown", fontsize=16)
    p.insert_image(fitz.Rect(72, 100, 540, 391), filename=str(chart))
    p.insert_textbox(fitz.Rect(72, 398, 540, 430),
                     "Figure 1: Revenue by region for FY2025, in millions of US dollars.", fontsize=9)
    p.insert_textbox(fitz.Rect(72, 450, 540, 760),
                     "Risks. The main risks for 2026 are semiconductor supply constraints, currency exposure to the "
                     "Japanese yen, and increased competition in autonomous mobile robots. Management mitigates "
                     "supply risk through dual sourcing of motor controllers.", fontsize=11)

    p = doc.new_page()  # image-only page: needs OCR or a VLM
    p.insert_image(p.rect, filename=str(scan))

    doc.save(path)


def safety_manual(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 80), "R7 Warehouse Robot - Safety Manual", fontsize=18)
    p.insert_textbox(fitz.Rect(72, 110, 540, 760), (
        "1. Battery handling. The R7 uses a 48 V lithium iron phosphate battery pack. Charge only with the "
        "approved DC-200 dock. Never charge below 0 °C or above 45 °C. Replace the pack after 3,000 cycles "
        "or if capacity drops below 70%.\n\n"
        "2. Emergency stop. Press the red emergency stop button on either side of the chassis. The robot "
        "halts within 0.4 seconds at full speed. Reset requires a supervisor key.\n\n"
        "3. Operating limits. Maximum payload is 600 kg. Maximum speed in shared zones is 1.2 m/s and "
        "2.0 m/s in robot-only zones. Keep aisles at least 1.5 m wide."
    ), fontsize=11)
    doc.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    chart, scan = OUT / "_chart.png", OUT / "_scan.png"
    bar_chart(chart)
    scanned_page(scan)
    files = [OUT / "acme_annual_report_2025.pdf", OUT / "r7_safety_manual.pdf", OUT / "remote_work_policy.md"]
    annual_report(files[0], chart, scan)
    safety_manual(files[1])
    files[2].write_text(
        "# Remote work policy\n\nEmployees may work remotely up to three days per week with manager approval. "
        "Core collaboration hours are 10:00 to 15:00 local time.\n\nThe company provides a one-time home office "
        "stipend of $750 and a monthly internet allowance of $40.\n", encoding="utf-8")
    print("Wrote:", *[f.relative_to(ROOT) for f in files], sep="\n  ")

    if args.ingest:
        from config import Config
        from docintel.pipeline import Pipeline

        pipe = Pipeline(Config)
        for f in files:
            doc = pipe.ingest_path(f, wait=True)
            doc = pipe.store.get_document(doc["id"])
            print(f"Indexed {doc['filename']}: {doc['status']} {doc['stats']} {doc.get('error') or ''}")


if __name__ == "__main__":
    main()
