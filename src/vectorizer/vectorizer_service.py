import json

import requests

from src.common.config.config import Config


class VectorizerService:
    def __init__(self, config: Config):
        self.config = config
        self.url = f"http://{config.get('VECTORIZER_HOST')}:{config.get('VECTORIZER_PORT')}/v1/embeddings"

    def embed(self, prompt: str) -> list[float]:
        client_cert = self.config.get("CLIENT_CERT")
        ca_cert = "onti-ca.crt"
        client_key = "DECFILE"

        data = {
            "input": prompt,
            "model": self.config.get("VECTORIZER_MODEL"),
            "encoding_format": "float"
        }

        try:
            with requests.post(
                self.url,
                json=data,
                cert=(client_cert, client_key),
                verify=ca_cert,
            ) as response:
                if response.status_code == 200:
                    return response.json()["data"][0]["embedding"]
                raise RuntimeError("Vectorizer ended not with 200: " + response.text)
        except BaseException as e:
            raise ConnectionError("Failed to call vectorizer: " + str(e))
