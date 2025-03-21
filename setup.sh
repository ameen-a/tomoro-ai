# Setup for ConvFinQA evaluation project

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # no color

echo -e "${GREEN}Starting setup for ConvFinQA evaluation project...${NC}"

# create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python -m venv .venv

# activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# create necessary directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/output

# create .env template if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env template...${NC}"
    echo "# openai api key" > .env
    echo "OPENAI_API_KEY=your_api_key_here" >> .env
    echo "# wandb setup" >> .env
    echo "WANDB_API_KEY=your_wandb_key_here" >> .env
    echo -e "${GREEN}Created .env template, please edit with your api keys${NC}"
else
    echo -e "${YELLOW}.env file already exists, skipping${NC}"
fi

# display success message
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To activate the environment in the future, run:${NC}"
echo "source .venv/bin/activate"
echo -e "${YELLOW}Remember to add your API keys to the .env file${NC}"

chmod +x setup.sh 