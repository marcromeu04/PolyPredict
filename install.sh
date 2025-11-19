#!/bin/bash
# PolyPredict Installation Script

echo "======================================"
echo "PolyPredict Installation"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Create virtual environment (optional but recommended)
echo ""
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "✓ Dependencies installed"

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw data/processed data/models data/results logs

echo "✓ Data directories created"

# Copy environment file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created (edit it to add your API keys)"
else
    echo "✓ .env file already exists"
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x run_tracker.py test_quick.py

echo "✓ Scripts are executable"

# Run quick test
echo ""
read -p "Run quick test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    echo "Running quick test..."
    python test_quick.py
fi

echo ""
echo "======================================"
echo "✓ Installation Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file to add API keys (optional)"
echo "  2. Run: python run_tracker.py"
echo "  3. Or open: notebooks/insider_tracker_demo.ipynb"
echo ""
