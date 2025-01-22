#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Failed to activate the virtual environment. Exiting..."
    exit 1
fi

# Install dependencies
python3 ../install.py

# Run the project to test
python3 _requirements.py

# Deactivate virtual environment when done
deactivate

# Remove the virtual environment
rm -r venv