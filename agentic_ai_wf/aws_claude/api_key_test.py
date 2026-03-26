import boto3 # (pip install boto3)
import os
from decouple import config # (pip install python-decouple)

os.environ["AWS_BEARER_TOKEN_BEDROCK"] = config("AWS_BEARER_TOKEN_BEDROCK")

# Create the Bedrock client
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Define the model and message
model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
messages = [{"role": "user", "content": [{"text": "Hello! Can you tell me about Amazon Bedrock?"}]}]

# Make the API call
response = client.converse(
    modelId=model_id,
    messages=messages,
)

# Print the response
print(response['output']['message']['content'][0]['text'])