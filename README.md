# lion

## install

```bash
pip install lion-os
```

## Features

- dynamic strutured output at runtime
- easily compose any multi-step process, manually or automatically
- use any model supported by `litellm`


## usage pattern

#### Using Branch.communicate

```python
instruction="Give me some ideas on fastapi coding questions"
context="we are hiring software engineers"
```

```python
from lion import iModel, Branch

imodel = iModel(
    model="openai/gpt-4o",
    api_key="OPENAI_API_KEY",
    temperature=0.2,
)

# create a branch
branch = Branch(imodel=imodel)

# communicate with AI models
result = await branch.communicate(
    instruction=instruction,
    context=context
)

print(result)
```

```plaintext
Certainly! Here are some concise FastAPI coding questions that you can use to assess the skills of software engineer candidates:

1. **Basic Endpoint Creation:**
   - Write a simple FastAPI application with a single GET endpoint that returns a JSON response with a message "Hello, World!".

...

```


#### Using Branch.operate

```python
from pydantic import BaseModel

class CodingQuestion(BaseModel):
    question: str
    evaluation_criteria: str

result = await branch.operate(
    instruction=instruction,
    context=context,
    operative_model=CodingQuestion,
)

result
```

```plaintext
CodingQuestion(question="Write a FastAPI application with a GET endpoint that returns a JSON response with a message 'Hello, World!'.", evaluation_criteria='Check for correct FastAPI setup, endpoint definition, and JSON response formatting.')
```


#### Using built-in operations

```python
from lion.operations import brainstorm

instruct={
    "instruction": instruction,
    "context": context,
}

result = await brainstorm(
    instruct=instruct,
    imodel=imodel,
    num_instruct=3,
    operative_model=CodingQuestion,
    auto_run=True,
    invoke_action=False,
)
```

```python
print("Number of ideas: ", len(result))
print(type(result[0]))
print(result[0].model_dump())
```

```plaintext
Number of ideas:  6
<class 'lion.core.models.CodingQuestion'>
{'question': "Develop a FastAPI endpoint '/user' that processes a POST request with a JSON payload. The payload should include 'first_name', 'last_name', and 'birth_year'. The endpoint must return a JSON response containing 'full_name' and 'age'.", 'evaluation_criteria': "The implementation should correctly use Pydantic models for input validation, handle errors for missing or invalid data gracefully, and accurately calculate the user's age based on the current year (2023). The code should be well-structured, readable, and follow FastAPI best practices.", 'instruct_models': [{'instruction': "Implement a FastAPI endpoint '/user' that accepts a POST request with a JSON payload containing 'first_name', 'last_name', and 'birth_year'. The endpoint should return a JSON response with 'full_name' and 'age'.", 'guidance': 'Use Pydantic models to validate the input data. Ensure the endpoint handles errors gracefully, such as missing or invalid data. Calculate the age based on the current year, 2023.', 'context': "The current year is 2023. Assume the input JSON payload will always contain 'first_name', 'last_name', and 'birth_year'.", 'reason': False, 'actions': False}]}
```

```python
instruct_models = []
for i in result:
    if i is not None:
        instruct_models.extend(getattr(i, "instruct_models", None) or [])

print("Number of ideas for next step: ", len(instruct_models))
print(type(instruct_models[0]))
print(instruct_models[0].model_dump())
```

```plaintext
Number of ideas for next step:  6
<class 'lion.protocols.operatives.instruct.InstructModel'>
{'instruction': "Implement a FastAPI endpoint '/user' that accepts a POST request with a JSON payload containing 'first_name', 'last_name', and 'birth_year'. The endpoint should return a JSON response with 'full_name' and 'age'.", 'guidance': 'Use Pydantic models to validate the input data. Ensure the endpoint handles errors gracefully, such as missing or invalid data. Calculate the age based on the current year, 2023.', 'context': "The current year is 2023. Assume the input JSON payload will always contain 'first_name', 'last_name', and 'birth_year'.", 'reason': False, 'actions': False}
```

### Requirements

python 3.11+ required
