import os
from openai import OpenAI
from config import settings


class PerplexityClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.perplexity_api_key,
            base_url=settings.perplexity_base_url
        )

    def get_client(self):
        return self.client


# Singleton instance
perplexity_client = PerplexityClient()