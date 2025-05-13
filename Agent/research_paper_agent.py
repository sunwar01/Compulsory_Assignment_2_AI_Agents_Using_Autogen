import ast
import copy
import logging
import json

from autogen import AssistantAgent, UserProxyAgent, ChatResult, register_function
from autogen.coding import LocalCommandLineCodeExecutor
from Tools.research_paper_search_tool import search_research_papers
from config import LLM_CONFIG

logging.getLogger("httpx").setLevel(logging.WARNING)

ReAct_prompt = """
Answer the following questions as best you can using only the provided tools.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do and which tool to use
Action: the action to take is ALWAYS search_research_papers
Action Input: the input to the action
Observation: the result of the action
Thought: I now know the final answer
Final Answer: the raw output from the search_research_papers tool as a JSON list of papers
TERMINATE

Call search_research_papers exactly once with appropriate parameters (e.g., topic, year, year_condition, min_citations). Output its raw JSON as the Final Answer. Do not reformat, summarize, or add text. Terminate with "TERMINATE" on a new line. If the tool returns an error or empty list, use that as the Final Answer.

Begin!
Question: {input}
"""

def react_prompt_message(context):
    return ReAct_prompt.format(input=context["question"])

def create_research_agent() -> AssistantAgent:
    return AssistantAgent(
        name="ResearchAssistant",
        system_message="""
        Use only the search_research_papers tool. Call it once and output exactly:
        Thought: I now know the final answer
        Final Answer: [tool output]
        TERMINATE
        Use the raw JSON output as the Final Answer, even if empty or an error. Do not reformat or add text.
        """,
        llm_config=copy.deepcopy(LLM_CONFIG)
    )

def create_critic_agent() -> AssistantAgent:
    return AssistantAgent(
        name="CriticAgent",
        system_message="You are a helpful AI assistant trained to critique other agents' research answers.",
        llm_config=copy.deepcopy(LLM_CONFIG),
    )

def create_user_proxy(code_executor: LocalCommandLineCodeExecutor):
    return UserProxyAgent(
        name="User",
        llm_config=None,
        is_termination_msg=lambda x: x.get("content", "") and (
            "terminate" in x.get("content", "").rstrip().lower() or
            "final answer:" in x.get("content", "").lower()
        ),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config={"executor": code_executor},
    )

def create_local_code_executor():
    return LocalCommandLineCodeExecutor(timeout=100)

def setup_agents():
    code_executor = create_local_code_executor()
    user_proxy = create_user_proxy(code_executor)
    research_agent = create_research_agent()
    critic_agent = create_critic_agent()

    register_function(
        search_research_papers,
        caller=research_agent,
        executor=user_proxy,
        name="search_research_papers",
        description="Search for research papers based on topic, publication year, and citation count."
    )

    return user_proxy, research_agent, critic_agent



def find_final_answer(chat_result: ChatResult):
    messages = chat_result.chat_history
    final_answer = None

    for message in reversed(messages):
        content = message.get("content", "")
        # Check for the tool response containing JSON
        if content.startswith("[{") and content.endswith("}]"):
            final_answer = content
            break
        elif content == "[]":
            final_answer = "[]"
            break
        # Check for explicit Final Answer block
        elif "Final Answer:" in content:
            final_answer_block = message.get("content", "")
            answer_block_lines = final_answer_block.split("\n")
            for line in answer_block_lines:
                if line.startswith("Final Answer:"):
                    final_answer = line.split("Final Answer:", 1)[1].strip()
                    # Verify it's valid JSON
                    try:
                        json.loads(final_answer)
                        break
                    except json.JSONDecodeError:
                        final_answer = None
                        continue

    return final_answer

def run_critic_on_output(critic_agent, user_prompt, agent_output):
    critic_prompt = f"""
You are evaluating the response of a research assistant agent based on the following criteria:

- Completeness (1-5): Does it fulfill the user's request?
- Relevance (1-5): Are the results aligned with the topic and conditions?
- JSON Correctness (1-5): Is the Final Answer valid raw JSON as instructed?
- Feedback: Explain why it was good or bad.


User Prompt:
{user_prompt}

Agent Output:
{agent_output}

Respond with a JSON object like:
{{
  "completeness": X,
  "relevance": X,
  "json_correctness": X,
  "feedback": "Brief explanation of what was good or bad."
}}
"""
    response = critic_agent.generate_reply([{"role": "user", "content": critic_prompt}])
    return response

# Example run
if __name__ == "__main__":
    user_proxy, research_agent, critic_agent = setup_agents()

    task = "Find available research papers on machine learning published in 2017 with a minimum of 250 citations."

    try:
        chat_result = user_proxy.initiate_chat(
            research_agent,
            message=task,
        )



        final_answer = find_final_answer(chat_result)
        if final_answer:
            print(f"\nFinal Answer:\n{final_answer}\n")

            # Critic Evaluation
            critic_response = run_critic_on_output(critic_agent, task, final_answer)

            try:
                print("Critic Evaluation:")
                print(critic_response)
            except Exception:
                print("Critic Evaluation:")
                print(critic_response)

        else:
            print("No final answer found.")

    except Exception as e:
        print(f"Error during chat: {e}")
        print("Tool calls made: Unknown due to error")
        print("Final Answer: Error during execution")
