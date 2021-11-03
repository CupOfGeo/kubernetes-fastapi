from fastapi import FastAPI
from transformers import GPTNeoForCausalLM, AutoTokenizer

from service.api.api_v1.api import router as api_router
from service.core.config import API_V1_STR, PROJECT_NAME
from service.core.models.output import GenOutPut
from service.core.models.input import PrompInput

import os
from google.cloud import storage


def read_file_blob(bucket_name, folder):
    # Instantiate a CGS client
    client = storage.Client()

    # The "folder" where the files you want to download are

    # Create this folder locally
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Retrieve all blobs with a prefix matching the folder
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder))

    for blob in blobs:
        if (not blob.name.endswith("/")):
            print(blob.name)
            #blob.download_to_filename(blob.name)


read_file_blob('central-bucket-george','Rapper')

path_to_model = '/Rapper'

# model = GPTNeoForCausalLM.from_pretrained(path_to_model).half().to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(path_to_model)

app = FastAPI(
    title=PROJECT_NAME,
    # if not custom domain
    # openapi_prefix="/prod"
)

app.include_router(api_router, prefix=API_V1_STR)


@app.get("/ping", summary="Check that the service is operational")
def pong():
    """
    Sanity check - this will let the user know that the service is operational.

    It is also used as part of the HEALTHCHECK. Docker uses curl to check that the API service is still running, by exercising this endpoint.

    """
    return {"ping": "pong!"}


@app.post("/generate", response_model=GenOutPut ,tags=["GPT-neo"])
def example_get(inputs: PrompInput):
    """
    Send promp, max_tokens, temp

    This will generate text from gpt-neo

    And this path operation will:
    * return gen_out
    """
    inputs.dict()
    text = inputs.text

    # ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    #
    # # add the length of the prompt tokens to match with the mesh-tf generation
    # max_length = inputs.max_tokens + ids.shape[1]
    #
    # gen_tokens = model.generate(
    #     ids,
    #     do_sample=True,
    #     min_length=max_length,
    #     max_length=max_length,
    #     temperature=inputs.temp,
    #     use_cache=True,
    #     num_return_sequences=3,
    # )
    # gen_text = tokenizer.batch_decode(gen_tokens)[0]

    gen_text = ", ".join(os.listdir('Rapper'))
    return {'out': gen_text}

