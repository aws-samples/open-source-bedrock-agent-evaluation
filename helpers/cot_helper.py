from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
import json


# Goal: Evaluate agent CoT using LLM-as-judge and output results
def evaluate_cot(
    agent_cot: str, agent_response: str, agent_info: list, client, MODEL_ID_EVAL_COT
):

    # Clean inputs to template
    agent_instructions = agent_info["agentInstruction"]
    collaborator_instructions = agent_info["collaborators"]
    # clean_agent_cot = agent_cot

    # Initialize Bedrock client
    llm = ChatBedrock(model=MODEL_ID_EVAL_COT, client=client)

    system_prompt_template = PromptTemplate(
        input_variables=[
            "agent_instructions",
            "collaborator_instructions",
            "agent_cot",
        ],
        template="""
        You are an expert evaluator analyzing AI Agent execution. Evaluate the agent's chain of thought performance on three key metrics:

        Agent Instructions:
        {agent_instructions}

        Collaboration Context:
        {collaborator_instructions}

        Agent Chain-of-Thought:
        {agent_cot}

        Final Agent Response:
        {agent_response}

        Evaluate on these three critical aspects:

        Helpfulness: How well does the execution satisfy explicit and implicit expectations?
        - Is it sensible, coherent, and clear?
        - Does it solve the task effectively?
        - Does it follow instructions?
        - Is it appropriately specific/general?
        - Does it anticipate user needs?

        Faithfulness: Does the execution stick to available information and context?
        - Does it avoid making unfounded claims?
        - Does it stay within the scope of given information?
        - Are conclusions properly supported?
        - Does it avoid contradicting the context?

        Instruction Following: Does it respect all explicit directions?
        - Follows specific requirements
        - Adheres to given constraints
        - Respects defined boundaries
        - Completes all requested steps

        Output your evaluation in the following Python dictionary format:

        {{
            "helpfulness": {{
                "score": <float 0-1>,
                "explanation": "<brief explanation>"
            }},
            
            "faithfulness": {{
                "score": <float 0-1>,
                "explanation": "<brief explanation>"
            }},
            
            "instruction_following": {{
                "score": <float 0-1>,
                "explanation": "<brief explanation>"
            }},
            
            "overall": {{
                "score": <float 0-1>,
                "explanation": "<brief summary>"
            }}
        }}

        Provide clear, concise explanations focusing on specific examples from the execution. Ensure your output is a valid Python dictionary that can be parsed directly. Do not include any text before or after the dictionary.
        """,
    )

    # Format the system prompt with function parameters
    system_prompt = system_prompt_template.format(
        agent_cot=agent_cot,
        agent_instructions=agent_instructions,
        collaborator_instructions=collaborator_instructions,
        agent_response=agent_response,
    )

    # Create the messages list
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "Please generate the chain-of-thought evaluation as specified.",
        },
    ]

    # Invoke the model to get CoT evaluation
    response = llm.invoke(messages)

    # Convert model response to dictionary
    eval_results = json.loads(response.content)
    eval_results["hallucination"] = {
        "score": 1 - eval_results["faithfulness"]["score"],
        "explanation": "Hallucination score derived as inverse of faithfulness, representing degree of unfounded or unsupported claims in response",
    }

    def clean_prompt_indentation(prompt_string):
        # Split into lines and strip leading/trailing whitespace
        lines = prompt_string.split("\n")
        cleaned_lines = [line.strip() for line in lines]
        # Rejoin with newlines
        return "\n".join(cleaned_lines)

    system_prompt = clean_prompt_indentation(system_prompt)

    return eval_results, system_prompt
