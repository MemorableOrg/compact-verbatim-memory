"""One client for OpenAI or Azure OpenAI. Azure is used when AZURE_OPENAI_API_KEY is set.

Env (in ~/.fm-env, never in the repo):
  AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (https://<resource>.openai.azure.com/),
  AZURE_OPENAI_API_VERSION (default 2024-12-01-preview),
  AZURE_OPENAI_DEPLOYMENTS, optional "gpt-5=my-gpt5-deployment,gpt-4o=my-4o" when deployment names differ from model names.
Scripts keep the model name for pricing and logs; only the API call sees the deployment name.
"""
import os
from openai import OpenAI, AzureOpenAI

_dep = dict(p.split("=", 1) for p in os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "").split(",") if "=" in p)
AZURE = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
ROUTER = os.environ.get("USE_OPENROUTER") == "1"

def client():
    if AZURE:
        return AzureOpenAI(api_key=os.environ["AZURE_OPENAI_API_KEY"], azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                           api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    if ROUTER:
        return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")
    return OpenAI()

def dep(model):
    if ROUTER: return model if "/" in model else f"openai/{model}"
    return _dep.get(model, model)
