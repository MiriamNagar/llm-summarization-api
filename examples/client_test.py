import requests
import json
import logging
import sys

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

url = "http://127.0.0.1:8000/summarize"
example_input_file = "basic_input.json"

try:
    # Load example payload
    logger.info("Loading example input from '%s'...", example_input_file)
    with open(f"examples/{example_input_file}", encoding="utf-8") as f:
        payload = json.load(f)

    logger.info("Sending request to API: %s", url)
    print()

    # Send POST request in streaming mode
    with requests.post(url, json=payload, stream=True) as response:
        if response.status_code != 200:
            logger.error("Error response: %s %s", response.status_code, response.text)
        else:
            logger.info("Streaming response:")
            print("-" * 50)

            # Iterate over streamed chunks
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)

            print("\n" + "-" * 50)
            logger.info("Streaming response completed.")

except Exception as e:
    logger.exception("Failed to send request or process response: %s", e)
