# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
python3 install.py

# Run the project to test
python3 _requirements.py

# If any modules are missing, install them and update requirements.txt
# pip install missing_module
# pip freeze > requirements.txt

# Deactivate virtual environment when done
deactivate

# Remove the virtual environment
rm -r venv