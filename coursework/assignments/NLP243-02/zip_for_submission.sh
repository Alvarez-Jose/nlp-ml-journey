#!/bin/bash

# Zip the files for submission
# Usage: ./zip_for_submission.sh


# If you have additional files, you need to add them to this script
if [ -f pyproject.toml ]; then
  zip -r hw2_submission.zip main.py model.py train.py download_best_model.py pyproject.toml hw2_report.pdf
elif [ -f requirements.txt ]; then
  zip -r hw2_submission.zip main.py model.py train.py download_best_model.py requirements.txt hw2_report.pdf
else
  echo "Error: No dependencies file found. Please provide either \`pyproject.toml\` or \`requirements.txt\`." >&2
  exit 1
fi

echo "Zipped files for submission: hw2_submission.zip"
