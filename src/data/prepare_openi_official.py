import argparse
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


OFFICIAL_URLS = {
    "reports": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
    "images": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
}

LABEL_KEYWORDS = {
    "Atelectasis": ["atelectasis"],
    "Cardiomegaly": ["cardiomegaly", "enlarged heart"],
    "Effusion": ["effusion", "pleural effusion"],
    "Infiltration": ["infiltration", "infiltrate"],
    "Mass": ["mass"],
    "Nodule": ["nodule"],
    "Pneumonia": ["pneumonia"],
    "Pneumothorax": ["pneumothorax"],
    "Consolidation": ["consolidation"],
    "Edema": ["edema", "oedema"],
    "Emphysema": ["emphysema"],
    "Fibrosis": ["fibrosis"],
    "Pleural": ["pleural thickening", "pleural"],
    "Hernia": ["hernia"],
}


def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)
    with output_path.open("rb") as f:
        header = f.read(256).lower()
    if b"<!doctype html" in header or b"<html" in header:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{url} returned an HTML page instead of an archive. "
            "OpenI may be under maintenance; retry later."
        )
    return output_path


def extract_tgz(archive_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir, filter="data")
    return output_dir


def text_of_abstract(root, label: str):
    values = []
    for node in root.findall(".//AbstractText"):
        if (node.attrib.get("Label") or "").lower() == label.lower() and node.text:
            values.append(node.text.strip())
    return " ".join(values)


def parse_report(xml_path: Path, image_root: Path):
    root = ET.parse(xml_path).getroot()
    indication = text_of_abstract(root, "INDICATION")
    findings = text_of_abstract(root, "FINDINGS")
    impression = text_of_abstract(root, "IMPRESSION")
    comparison = text_of_abstract(root, "COMPARISON")
    report = " ".join(part for part in [indication, findings, impression] if part).strip()

    mesh_terms = " ".join(node.text or "" for node in root.findall(".//MeSH/*"))
    problems = " ".join(node.text or "" for node in root.findall(".//Problems/*"))
    label_source = f"{mesh_terms} {problems}".lower()

    labels = {
        label: int(any(keyword in label_source for keyword in keywords))
        for label, keywords in LABEL_KEYWORDS.items()
    }

    rows = []
    for image_node in root.findall(".//parentImage"):
        image_id = image_node.attrib.get("id")
        if not image_id:
            continue
        image_path = image_root / f"{image_id}.png"
        rows.append(
            {
                "uid": xml_path.stem,
                "image_id": image_id,
                "image_path": str(image_path),
                "report": report,
                "indication": indication,
                "findings": findings,
                "impression": impression,
                "comparison": comparison,
                **labels,
            }
        )
    return rows


def find_xml_dir(root: Path):
    xml_files = list(root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML reports found under {root}")
    return xml_files


def build_csv(data_dir: Path, output_csv: Path):
    reports_dir = data_dir / "reports"
    images_dir = data_dir / "images"
    xml_files = find_xml_dir(reports_dir)
    rows = []
    for xml_path in xml_files:
        rows.extend(parse_report(xml_path, images_dir))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No OpenI rows were parsed from the XML reports.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv, len(df)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare the official OpenI/IU X-Ray dataset.")
    parser.add_argument("--data-dir", default="data/openi", help="Directory used for OpenI archives and extracted files.")
    parser.add_argument("--output-csv", default="data/openi/openi_prepared.csv")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded archives/extracted files.")
    parser.add_argument("--reports-only", action="store_true", help="Download only XML reports; image paths are still written.")
    parser.add_argument("--images-only", action="store_true", help="Download/extract images without rebuilding reports first.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    archives_dir = data_dir / "archives"
    reports_archive = archives_dir / "NLMCXR_reports.tgz"
    images_archive = archives_dir / "NLMCXR_png.tgz"

    if not args.skip_download:
        if not args.images_only:
            download_file(OFFICIAL_URLS["reports"], reports_archive)
        if not args.reports_only:
            download_file(OFFICIAL_URLS["images"], images_archive)

    reports_exist = (data_dir / "reports").exists()
    images_exist = (data_dir / "images").exists()

    if reports_archive.exists() and not args.images_only and not reports_exist:
        extract_tgz(reports_archive, data_dir / "reports")
    if images_archive.exists() and not images_exist:
        extract_tgz(images_archive, data_dir / "images")

    if not args.images_only:
        output_csv, n_rows = build_csv(data_dir, Path(args.output_csv))
        print(f"Wrote {n_rows} OpenI image-report rows to {output_csv}")


if __name__ == "__main__":
    main()
