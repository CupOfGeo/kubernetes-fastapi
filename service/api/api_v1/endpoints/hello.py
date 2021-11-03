
from fastapi import APIRouter

from service.core.models.output import MessageOutput, GenOutPut
from service.core.models.input import MessageInput, PrompInput
from service.core.logic.business_logic import run_prime_factor_calculation
import os



router = APIRouter()




@router.post("/hello", response_model=MessageOutput, tags=["hello post"])
def hello_endpoint(inputs: MessageInput):
    """
    Respond to requests on the hello endpoint

    """

    n, largest_prime_factor, elapsed_time = run_prime_factor_calculation()

    return {
        "message1": "Hello, world!",
        "message2": f"The largest prime factor of {n} is {largest_prime_factor}. Calculation took {elapsed_time:0.3f} seconds.",
        "n": n,
        "largest_prime_factor": largest_prime_factor,
        "elapsed_time": elapsed_time,
    }
