import os

import pandas as pd
from lxml import etree
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = "./data"
XML_DIR = os.path.join(DATA_DIR, "reports")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_CSV = "./openi_processed_labels.csv"


def parse_report(xml_file):
    """Parses a single XML report to extract text and labels."""
    try:
        tree = etree.parse(xml_file)
        root = tree.getroot()
        parent_images = root.findall("parentImage")

        if not parent_images:  # No parent imagea found
            return None

        image_id = parent_images[0].get("id")  # Only take one image for demo purposes

        image_path = None
        for fname in os.listdir(IMAGE_DIR):
            if fname.startswith(image_id):
                image_path = os.path.join(IMAGE_DIR, fname)
                break

        if not image_path:
            return None  # Skip if no matching image is found

        # Extract the 'FINDINGS' text
        findings_node = root.find('.//AbstractText[@Label="FINDINGS"]')
        report_text = (
            findings_node.text
            if findings_node is not None and findings_node.text
            else ""
        )

        # Extract MeSH terms as labels
        mesh_terms = [term.text.lower() for term in root.findall(".//MeSH/major")]

        # We need both text and labels to include the record
        if not report_text or not mesh_terms:
            return None

        return {
            "image_path": image_path,
            "report_text": report_text.strip(),
            "labels": mesh_terms,
        }
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None


# --- Main Script ---
if __name__ == "__main__":
    xml_files = [
        os.path.join(XML_DIR, f) for f in os.listdir(XML_DIR) if f.endswith(".xml")
    ]

    # Process all XML files with a progress bar
    print("Parsing XML reports...")
    all_data = []
    for xml_file in tqdm(xml_files, desc="Processing files"):
        parsed_data = parse_report(xml_file)
        if parsed_data:
            all_data.append(parsed_data)

    # Convert to a DataFrame
    df = pd.DataFrame(all_data)
    print(f"\nSuccessfully parsed {len(df)} records with images, text, and labels.")

    # Now, process the labels into a machine-readable format
    print("Binarizing labels...")
    mlb = MultiLabelBinarizer()

    # Fit and transform the labels
    binarized_labels = mlb.fit_transform(df["labels"])

    # Create a new DataFrame for the binarized labels
    labels_df = pd.DataFrame(binarized_labels, columns=mlb.classes_, index=df.index)

    # Combine the original DataFrame (without the raw labels) with the new binarized labels
    final_df = pd.concat([df[["image_path", "report_text"]], labels_df], axis=1)

    # Save the final processed data to a CSV file
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nProcessing complete! Data saved to {OUTPUT_CSV}")
    print("\nHere's a preview of your labeled data:")
    print(final_df.head())

    print("\nAvailable labels (columns):")
    print(list(final_df.columns[2:]))
