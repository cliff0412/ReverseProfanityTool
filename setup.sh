#!/bin/bash

# Check for Python 3 and pip
if command -v python3 &>/dev/null && command -v pip3 &>/dev/null; then
    echo "Python 3 and pip are installed"
else
    echo "Python 3 or pip is not installed"
    exit 1
fi

# Set up Python virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install Python requirements
pip3 install -r requirements.txt

# Check if Go is installed
if ! command -v go &>/dev/null; then
    echo "Go is not installed. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt update
        sudo apt install golang-go
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install go
    else
        echo "Unsupported OS for Go installation script"
        exit 1
    fi
fi

# Assuming the Go files are named "script1.go" and "script2.go"
go build script1.go
./script1

go build script2.go
./script2

# Deactivate Python virtual environment
deactivate
