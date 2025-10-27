#!/bin/bash

# Zip the files for submission
# Usage: ./zip_for_submission.sh


# If you have additional files, you need to add them to this script
if [ -f pyproject.toml ]; then
  zip -r hw1_submission.zip model.py train.py vectorizer.joblib best_model.pt pyproject.toml requirements.txt hw1_report.pdf
elif [ -f requirements.txt ]; then
  zip -r hw1_submission.zip model.py train.py vectorizer.joblib best_model.pt requirements.txt hw1_report.pdf
else
  echo "Error: No dependencies file found. Please provide either \`pyproject.toml\` or \`requirements.txt\`." >&2
  exit 1
fi

echo "Zipped files for submission: hw1_submission.zip"
