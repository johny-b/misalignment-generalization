from pathlib import Path
from dotenv import load_dotenv

project_dir = Path(__file__).parent.parent.resolve()
load_dotenv(project_dir / ".env")

def load_env_variables() -> None:
    load_dotenv(project_dir / ".env")
