import os
from pathlib import Path
from dotenv import load_dotenv

project_dir = Path(__file__).parent
load_dotenv(project_dir / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
