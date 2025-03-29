import time
from datetime import datetime
from typing import Dict, Any, Tuple
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToolCorrectnessMetric
from evaluators.cot_evaluator import ToolEvaluator
from helpers.deepeval_helper import DeepevalHelper
from helpers.tool_usage_extractor import ToolUsageExtractor


class ToolCorrectnessEvaluator(ToolEvaluator):
    def __init__(self, **kwargs):
        """
        Initialize Tool Correctness Evaluator with all necessary components

        Args:
            **kwargs: Arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.tool_usage_extractor = ToolUsageExtractor()
        self.deepeval_helper = DeepevalHelper()

    def _initialize_clients(self) -> None:
        """Initialize evaluation-specific models using shared clients"""
        # Use shared clients
        self.bedrock_agent_client = self.clients["bedrock_agent_client"]
        self.bedrock_agent_runtime_client = self.clients["bedrock_agent_runtime"]
        self.bedrock_client = self.clients["bedrock_runtime"]

    def invoke_agent(self, tries: int = 1) -> Tuple[Dict[str, Any], datetime]:
        """
        Invoke the agent and process its response with retry logic

        Args:
            tries (int): Number of retry attempts

        Returns:
            Tuple of (full_trace, processed_response, start_time)
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
                "question": self.question,
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
        Evaluate tool correctness using DeepEval's ToolCorrectnessMetric

        Args:
            metadata: Dictionary containing evaluation metadata

        Returns:
            Dictionary containing evaluation results
        """
        try:
            tool_usage_data = metadata.get("tool_usage", {})

            if not tool_usage_data.get("tool_calls", []):
                return {
                    "metrics_scores": {
                        "tool_correctness": {
                            "score": 0.0,
                            "explanation": "No tool calls found in the agent trace",
                        }
                    }
                }

            # Convert ground truth and actual tool calls to DeepEval format
            expected_tool_calls = DeepevalHelper.convert_to_deepeval_tool_calls(
                self.ground_truth["expected_tool_calls"]
            )
            actual_tool_calls = (
                DeepevalHelper.convert_bedrock_format_to_deepeval_tool_calls(
                    tool_usage_data.get("tool_calls", [])
                )
            )

            test_case = LLMTestCase(
                input=self.question,
                actual_output=metadata.get("agent_response"),
                # Replace this with the tools that was actually used by your LLM agent
                tools_called=actual_tool_calls,
                expected_tools=expected_tool_calls,
            )

            metric = ToolCorrectnessMetric()

            metric.measure(test_case)
            score = metric.score
            reason = metric.reason

            agent_metrics = {}
            for agent_id, agent_data in tool_usage_data.get("agents", {}).items():
                agent_name = agent_data.get("agent_name", f"Agent-{agent_id}")
                tool_calls = agent_data.get("tool_calls", [])

                agent_metrics[agent_name] = {
                    "tool_calls_count": len(tool_calls),
                    "tools_used": list(
                        set(call.get("tool_name", "") for call in tool_calls)
                    ),
                }

            return {
                "metrics_scores": {
                    "tool_correctness": {
                        "score": float(score),
                        "explanation": reason
                        or f"Tool correctness score across {len(tool_usage_data.get('tool_calls', []))} tool calls",
                    }
                },
                "agent_metrics": agent_metrics,
                "tool_usage_data": tool_usage_data,
            }

        except Exception as e:
            raise Exception(f"Error evaluating tool correctness: {str(e)}")
