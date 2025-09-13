import time # added
from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from google import genai
from configs.key import *

def create_message(prompt):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages

def create_client(model_name):
    if "gpt" in model_name:
        client = OpenAI(api_key=OpenAI_API_KEY)
    elif "gemini" in model_name:
        client = genai.Client(api_key=GEMINI_API_KEY)
    elif "changan" in model_name:
        client = ChatBot()
    else:
        raise ValueError("Invalid model name!")
    return client
    
def get_response(client, prompt, model, temperature=1, max_tokens=1000):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    messages = create_message(prompt)

    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            if "gpt" in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # print(response)
                rslt = response.choices[0].message.content
            elif "gemini" in model:
                response = client.models.generate_content(
                    model=model, contents=prompt,
                    # config=genai.types.GenerateContentConfig(
                    #         max_output_tokens=args.max_tokens,
                    #         temperature=args.temperature,
                    #     ),
                )
                rslt = response.text
            elif "changan" in model:
                rslt = client.ask(prompt)
            else:
                raise ValueError("Invalid model name!")
            success = True
        except RateLimitError as e:
            print(f"sleep {SLEEP_TIME} seconds for rate limit error")
            time.sleep(SLEEP_TIME)
        except APITimeoutError as e:
            print(f"sleep {SLEEP_TIME} seconds for time out error")
            time.sleep(SLEEP_TIME)
        except APIConnectionError as e:
            print(f"sleep {SLEEP_TIME} seconds for api connection error")
            time.sleep(SLEEP_TIME)
        except APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except genai.errors.APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt