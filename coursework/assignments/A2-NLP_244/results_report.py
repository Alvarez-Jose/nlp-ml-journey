import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Load all results
with open('all_predictions.json', 'r') as f:
    predictions = json.load(f)
    
with open('error_analysis_summary.json', 'r') as f:
    error_summary = json.load(f)

# Generate HTML report
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Homework 2 Results Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .summary {{ background-color: #e8f4f8; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Homework 2: NER Model Results</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>Summary Statistics</h2>
        <p>Total predictions: {error_summary['total_predictions']}</p>
        <p>Number of files processed: {len(predictions)}</p>
    </div>
    
    <h2>Predictions per File</h2>
    <table>
        <tr>
            <th>Filename</th>
            <th>Number of Predictions</th>
            <th>Preview</th>
        </tr>
"""

for filename, preds in predictions.items():
    preview = ', '.join([f"{p['tag']}: {p['text'][:20]}" for p in preds[:3]])
    html_content += f"""
        <tr>
            <td>{filename}</td>
            <td>{len(preds)}</td>
            <td>{preview}...</td>
        </tr>
    """

html_content += """
    </table>
    
    <h2>Tag Distribution</h2>
    <table>
        <tr>
            <th>Tag</th>
            <th>Count</th>
        </tr>
"""

for tag, count in error_summary['tag_distribution'].items():
    html_content += f"""
        <tr>
            <td>{tag}</td>
            <td>{count}</td>
        </tr>
    """

html_content += """
    </table>
    
    <h2>Sample Predictions</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Tag</th>
            <th>Text</th>
        </tr>
"""

# Show sample predictions (first 20)
sample_count = 0
for filename, preds in predictions.items():
    for pred in preds[:5]:  # First 5 from each file
        if sample_count < 20:
            html_content += f"""
        <tr>
            <td>{filename}</td>
            <td>{pred['tag']}</td>
            <td>{pred['text']}</td>
        </tr>
            """
            sample_count += 1

html_content += """
    </table>
</body>
</html>
"""

with open('results_report.html', 'w') as f:
    f.write(html_content)

print("Results report generated: results_report.html")

# Create final submission zip
import zipfile
import os

files_to_zip = [
    'train.json',
    'validation.json',
    'file_contents.json',
    'ner_model.pt',
    'all_predictions.json',
    'error_analysis_summary.json',
    'results_report.html',
    'training_curves.png',
    'error_analysis_distribution.png'
]

with zipfile.ZipFile('homework2_submission.zip', 'w') as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file)
            print(f"Added {file} to submission zip")

print("\nSubmission package created: homework2_submission.zip")