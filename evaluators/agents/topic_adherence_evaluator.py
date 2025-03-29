import time
from datetime import datetime
from typing import Dict, Any, List, Tuple
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import TopicAdherenceScore
from ragas.dataset_schema import MultiTurnSample
from langchain_aws.chat_models import ChatBedrock
from evaluators.cot_evaluator import ToolEvaluator
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall


class TopicAdherenceEvaluator(ToolEvaluator):
    def __init__(self, **kwargs):
        """
        Initialize Topic Adherence Evaluator with all necessary components

        Args:
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)

    def _initialize_clients(self) -> None:
        """Initialize evaluation-specific models using shared clients"""
        self.bedrock_agent_client = self.clients["bedrock_agent_client"]
        self.bedrock_agent_runtime_client = self.clients["bedrock_agent_runtime"]
        self.bedrock_client = self.clients["bedrock_runtime"]

        self.llm_for_evaluation = ChatBedrock(
            model_id=self.config["MODEL_ID_EVAL"],
            max_tokens=100000,
            client=self.bedrock_client,  # Use shared client
        )

    def _convert_trace_to_messages(
        self, full_trace: List[Dict], agent_answer: str
    ) -> List[Any]:
        """
        Convert the agent trace into RAGAS message format

        Args:
            full_trace: List of trace events
            agent_answer: Final agent response

        Returns:
            List of RAGAS messages
        """
        messages = []

        messages.append(HumanMessage(content=str(self.question)))

        for event in full_trace:
            if "trace" in event:
                trace_obj = event["trace"]["trace"]
                if "orchestrationTrace" in trace_obj:
                    orc_trace = trace_obj["orchestrationTrace"]

                    # Handle tool calls
                    if "invocationInput" in orc_trace:
                        invoc_input = orc_trace["invocationInput"]
                        if "actionGroupInvocationInput" in invoc_input:
                            action_input = invoc_input["actionGroupInvocationInput"]
                            tool_call = ToolCall(
                                name=str(action_input.get("function", "")),
                                args={
                                    k: str(v)
                                    for k, v in action_input.get(
                                        "parameters", {}
                                    ).items()
                                },
                            )
                            messages.append(
                                AIMessage(
                                    content="Using tool to process request",
                                    tool_calls=[tool_call],
                                )
                            )

                    # Handle tool outputs
                    if "observation" in orc_trace:
                        obs = orc_trace["observation"]
                        if "actionGroupInvocationOutput" in obs:
                            tool_output = str(
                                obs["actionGroupInvocationOutput"].get("text", "")
                            )
                            messages.append(ToolMessage(content=tool_output))

        if agent_answer:
            messages.append(AIMessage(content=str(agent_answer)))

        print(f"Messages: {messages}")

        return messages

    def invoke_agent(self, tries: int = 1) -> Tuple[Dict[str, Any], datetime]:
        """
        Invoke the agent and process its response

        Args:
            tries (int): Number of retry attempts

        Returns:
            Tuple of (processed_response, start_time)
        """
        agent_start_time = datetime.now()
        max_retries = 3

        try:
            raw_response = self.bedrock_agent_runtime_client.invoke_agent(
                inputText=self.question,
                agentId=self.config["AGENT_ID"],
                agentAliasId=self.config["AGENT_ALIAS_ID"],
                sessionId=self.session_id,
                enableTrace=self.config["ENABLE_TRACE"],
            )

            agent_answer = None
            input_tokens = 0
            output_tokens = 0
            full_trace = []

            for event in raw_response["completion"]:
                if "chunk" in event:
                    agent_answer = event["chunk"]["bytes"].decode("utf-8")

                elif "trace" in event:
                    full_trace.append(event["trace"])
                    trace_obj = event["trace"]["trace"]

                    if "orchestrationTrace" in trace_obj:
                        orc_trace = trace_obj["orchestrationTrace"]

                        # Extract token usage
                        if "modelInvocationOutput" in orc_trace:
                            usage = orc_trace["modelInvocationOutput"]["metadata"][
                                "usage"
                            ]
                            input_tokens += usage.get("inputTokens", 0)
                            output_tokens += usage.get("outputTokens", 0)

            processed_response = {
                "agent_generation_metadata": {
                    "ResponseMetadata": raw_response.get("ResponseMetadata", {}),
                },
                "agent_answer": agent_answer,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "full_trace": full_trace,
            }

            return full_trace, processed_response, agent_start_time

        except Exception as e:
            if (
                hasattr(e, "response")
                and "Error" in e.response
                and e.response["Error"].get("Code") == "throttlingException"
                and tries <= max_retries
            ):

                wait_time = 30 * tries
                print(
                    f"Throttling occurred. Attempt {tries} of {max_retries}. "
                    f"Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)
                return self.invoke_agent(tries + 1)
            else:
                raise e

    async def evaluate_response(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate topic adherence using RAGAS

        Args:
            metadata (Dict[str, Any]): Evaluation metadata containing:
                - question: The user's question
                - ground_truth: Dictionary containing:
                    - topics: List of topics to evaluate against
                - agent_response: The agent's response
                - full_trace: Full trace of agent interaction

        Returns:
            Dict containing evaluation results
        """
        try:
            topics = [
                str(topic) for topic in metadata["ground_truth"].get("topics", [])
            ]
            full_trace = metadata.get("full_trace", [])
            agent_response = metadata.get("agent_response", "")

            if not topics:
                raise ValueError("No topics provided in ground truth for evaluation")

            # Convert trace to RAGAS message format
            messages = self._convert_trace_to_messages(full_trace, agent_response)

            sample = MultiTurnSample(user_input=messages, reference_topics=topics)

            scorer = TopicAdherenceScore(
                llm=LangchainLLMWrapper(self.llm_for_evaluation), mode="precision"
            )

            adherence_score = await scorer.multi_turn_ascore(sample)

            metrics = {
                "topic_adherence": {
                    "score": float(adherence_score),
                    "explanation": f"Response adherence to topics: {', '.join(topics)}",
                }
            }

            return {
                "metrics_scores": metrics,
                "detailed_analysis": {
                    "evaluated_topics": topics,
                    "conversation_turns": len(messages),
                },
            }

        except Exception as e:
            raise Exception(f"Error evaluating topic adherence: {str(e)}")
