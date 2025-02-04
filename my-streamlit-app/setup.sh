#!/bin/bash

# Update package list and install pip if not installed
sudo apt-get update
sudo apt-get install -y python3-pip

# Install required Python packages
pip3 install -r requirements.txt

# Run the Streamlit app
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0