from rag.logger import get_logger
import yaml
from dotenv import load_dotenv
import os


def load_conf(conf_path="config/config.yaml"):
    logger = get_logger(__name__)
    with open(conf_path, "r") as f:
        config = yaml.safe_load(f)

    env_file = config.get("env_file", ".env")
    load_dotenv(env_file)
    if os.getenv("OPENAI_API_KEY") is not None:
        logger.info("OPENAI_API_KEY found.")
    else:
        logger.critical("OPENAI_API_KEY environment variable not found")
    return config
