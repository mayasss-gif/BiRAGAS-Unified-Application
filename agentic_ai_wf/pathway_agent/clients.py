from openai import OpenAI
from google import genai
from .config import settings
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig


openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)


gemini_client =  AsyncOpenAI(
    api_key=settings.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=gemini_client
)

run_config = RunConfig(
    model=gemini_model,
    model_provider=gemini_client,
    # model_settings=ModelSettings(
    #     max_tokens=4096,
    #     temperature=0.5,
    #     top_p=1,
    # ),
    tracing_disabled=True
)