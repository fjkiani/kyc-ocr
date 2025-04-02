#!/bin/bash
# Document Processing System Launcher Script

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=======================================================${NC}"
echo -e "${YELLOW}      Document Processing System Launcher              ${NC}"
echo -e "${YELLOW}=======================================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}Python 3.8 or higher is required.${NC}"
    echo -e "${RED}Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Python $PYTHON_VERSION detected.${NC}"

# Check for system dependencies
echo -e "${YELLOW}Checking system dependencies...${NC}"

# Check for libmagic
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! brew list libmagic &> /dev/null; then
        echo -e "${YELLOW}libmagic not found. Installing...${NC}"
        brew install libmagic
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install libmagic. File type detection may not work correctly.${NC}"
            echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
            read -r response
            if [[ "$response" != "y" ]]; then
                exit 1
            fi
        else
            echo -e "${GREEN}libmagic installed successfully.${NC}"
        fi
    else
        echo -e "${GREEN}libmagic is already installed.${NC}"
    fi
    
    # Check for poppler
    if ! brew list poppler &> /dev/null; then
        echo -e "${YELLOW}poppler not found. Installing...${NC}"
        brew install poppler
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install poppler. PDF processing may not work correctly.${NC}"
            echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
            read -r response
            if [[ "$response" != "y" ]]; then
                exit 1
            fi
        else
            echo -e "${GREEN}poppler installed successfully.${NC}"
        fi
    else
        echo -e "${GREEN}poppler is already installed.${NC}"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! dpkg -l libmagic1 &> /dev/null; then
        echo -e "${YELLOW}libmagic not found. Installing...${NC}"
        sudo apt-get update && sudo apt-get install -y libmagic1
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install libmagic. File type detection may not work correctly.${NC}"
            echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
            read -r response
            if [[ "$response" != "y" ]]; then
                exit 1
            fi
        else
            echo -e "${GREEN}libmagic installed successfully.${NC}"
        fi
    else
        echo -e "${GREEN}libmagic is already installed.${NC}"
    fi
    
    # Check for poppler
    if ! dpkg -l poppler-utils &> /dev/null; then
        echo -e "${YELLOW}poppler-utils not found. Installing...${NC}"
        sudo apt-get update && sudo apt-get install -y poppler-utils
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install poppler-utils. PDF processing may not work correctly.${NC}"
            echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
            read -r response
            if [[ "$response" != "y" ]]; then
                exit 1
            fi
        else
            echo -e "${GREEN}poppler-utils installed successfully.${NC}"
        fi
    else
        echo -e "${GREEN}poppler-utils is already installed.${NC}"
    fi
else
    echo -e "${YELLOW}Automatic installation of system dependencies is not supported on this OS.${NC}"
    echo -e "${YELLOW}Please ensure libmagic and poppler are installed manually.${NC}"
    echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated.${NC}"

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Checking dependencies...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install dependencies.${NC}"
        echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
        read -r response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}Dependencies installed.${NC}"
    fi
fi

# Create temp directory if it doesn't exist
if [ ! -d "temp" ]; then
    echo -e "${YELLOW}Creating temp directory...${NC}"
    mkdir -p temp
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create temp directory.${NC}"
        echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
        read -r response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}Temp directory created.${NC}"
    fi
fi

# Check if test directory has sample images
if [ ! -d "test" ] || [ -z "$(ls -A test/*.jpg test/*.jpeg test/*.png 2>/dev/null)" ]; then
    echo -e "${YELLOW}No sample images found. Downloading sample images...${NC}"
    if [ -f "download_sample_images.py" ]; then
        python download_sample_images.py
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download sample images.${NC}"
            echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
            read -r response
            if [[ "$response" != "y" ]]; then
                exit 1
            fi
        else
            echo -e "${GREEN}Sample images downloaded successfully.${NC}"
        fi
    else
        echo -e "${RED}Sample image downloader script not found.${NC}"
        echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
        read -r response
        if [[ "$response" != "y" ]]; then
            exit 1
        fi
    fi
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating one...${NC}"
    echo -e "${YELLOW}Please enter your Fireworks API key:${NC}"
    read -r api_key
    echo "FIREWORKS_API_KEY=$api_key" > .env
    echo -e "${GREEN}.env file created.${NC}"
fi

# Run the application
echo -e "${YELLOW}=======================================================${NC}"
echo -e "${YELLOW}      Starting Document Processing System              ${NC}"
echo -e "${YELLOW}=======================================================${NC}"

# Run with Python if run_app.py exists, otherwise use streamlit directly
if [ -f "run_app.py" ]; then
    python run_app.py
elif [ -f "app.py" ]; then
    streamlit run app.py
else
    echo -e "${RED}No app.py or run_app.py found.${NC}"
    exit 1
fi 