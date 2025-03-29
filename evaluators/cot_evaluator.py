import re
import time
import json
from datetime import datetime
from langfuse import Langfuse
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import helpers.cot_helper as cot_helper


from helpers.tool_usage_extractor import ToolUsageExtractor


class ToolEvaluator(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        agent_info: Dict[str, Any],
        eval_type: str,
        question: str,
        ground_truth: Any,
        trace_id: str,
        session_id: str,
        question_id: int,
        trajectory_id: str,
    ):
        """
        Base class for tool evaluation

        Args:
            config (Dict[str, Any]): Configuration dictionary containing credentials and settings
            agent_info (Dict[str, Any]): Information about the agent being evaluated
            eval_type (str): Type of evaluation being performed
            question (str): Question to evaluate
            ground_truth (Any): Ground truth answer
            trace_id (str): Unique identifier for the evaluation trace
            session_id (str): Identifier for the evaluation session
            question_id (int): Identifier for the specific question
        """
        self.config = config
        self.agent_info = agent_info
        self.eval_type = eval_type
        self.question = question
        self.ground_truth = ground_truth
        self.trace_id = trace_id
        self.session_id = session_id
        self.question_id = question_id
        self.trajectory_id = trajectory_id
        self.clients = config.get("clients", {})
        self.langfuse = Langfuse()
        self.tool_usage_extractor = ToolUsageExtractor()

        self._initialize_clients()

    @abstractmethod
    def _initialize_clients(self) -> None:
        """Initialize tool-specific clients"""
        pass

    @abstractmethod
    def invoke_agent(self, tries: int = 1) -> Tuple[Dict[str, Any], datetime]:
        """
        Invoke the specific tool and process its response

        Args:
            tries (int): Number of retry attempts

        Returns:
            Tuple containing processed response and start time
        """
        pass

    @abstractmethod
    async def evaluate_response(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate tool response using specified metrics

        Args:
            metadata (Dict[str, Any]): Metadata for evaluation

        Returns:
            Dict containing evaluation results
        """
        pass

    def _add_agent_collaborators(self, agents_used, trimmed_orc_trace):

        for item in trimmed_orc_trace:
            # Check for invocationInput
            if "invocationInput" in item:
                if "agentCollaboratorInvocationInput" in item["invocationInput"]:
                    name = item["invocationInput"]["agentCollaboratorInvocationInput"][
                        "agentCollaboratorName"
                    ]
                    agents_used.add(name)

            # Check for observation with agentCollaboratorInvocationOutput
            if "observation" in item:
                if "agentCollaboratorInvocationOutput" in item["observation"]:
                    name = item["observation"]["agentCollaboratorInvocationOutput"][
                        "agentCollaboratorName"
                    ]
                    agents_used.add(name)

        return agents_used

        # print("Collaborators Used for Question: {}".format(collaborator_names))

    def _create_trace(self) -> Any:
        """Create and initialize a Langfuse trace"""
        traj_num = re.findall(r"\d+$", self.trajectory_id)[0]

        return self.langfuse.trace(
            id=self.trace_id,
            session_id=self.session_id,
            input=self.question,
            name=f"T{traj_num}-Q{self.question_id}-{self.eval_type}",
            user_id=self.config["AGENT_ID"],
            tags=[
                self.eval_type,
                self.agent_info["agentModel"],
                self.agent_info["agentType"],
            ],
        )

    def _handle_error(self, trace: Any, error: Exception, stage: str) -> None:
        """Handle and log errors during evaluation without raising"""
        traj_num = re.findall(r"\d+$", self.trajectory_id)[0]

        error_message = f"{stage} error: {str(error)}"
        trace.update(
            name=f"[ERROR] T{traj_num}-Q{self.question_id}-{self.eval_type}",
            metadata={"errorMessage": error_message},
            output={"Agent Error": error_message},
            tags=["ERROR"],
        )
        print(f"Error in {stage}: {error_message}")

    def process_trace_step(self, trace_step):

        if "orchestrationTrace" in trace_step:
            # print("This is an orchestration trace")

            trace = trace_step["orchestrationTrace"]

            orchestration_trace = {
                "name": "Orchestration",
                "input": json.loads(trace["modelInvocationInput"].get("text", "{}")),
                "output": json.loads(
                    trace["modelInvocationOutput"]["rawResponse"].get("content", "{}")
                ),
                "metadata": trace["modelInvocationOutput"].get("metadata", {}),
                "trace_id": trace["modelInvocationInput"].get("traceId"),
            }

            if "observation" in trace and "finalResponse" in trace["observation"]:
                orchestration_trace["final_response"] = {
                    "text": trace["observation"]["finalResponse"].get("text")
                }

            # print("Data: {}".format(orchestration_trace))
            return orchestration_trace

    def combine_traces(self, full_trace):

        trace_ids = []
        trace_steps = []
        cur_dict = {}

        def find_trace_id(data):
            if isinstance(data, dict):
                # If traceId is directly in this dictionary, return it
                if "traceId" in data:
                    return data["traceId"]
                # Otherwise search through all values in the dictionary
                for value in data.values():
                    result = find_trace_id(value)
                    if result:
                        return result
            # If the value is a list, search through its elements
            elif isinstance(data, list):
                for item in data:
                    result = find_trace_id(item)
                    if result:
                        return result
            return None

        # iterate through all the traces
        for cur_trace in full_trace:

            cur_trace_id = find_trace_id(cur_trace)
            # LOGIC FOR INITIALIZING NEW DICTIONARY
            # only for the first instsance of a single trace ID
            if cur_trace_id not in trace_ids:
                # print("Unique trace ID: {}".format(cur_trace_id))
                # initialize new dict with the agent information
                if cur_dict:
                    trace_steps.append(cur_dict)
                    cur_dict = {}

                cur_dict = {
                    key: value for key, value in cur_trace.items() if key != "trace"
                }

                # print("Unique dict: {}".format(cur_dict))
                trace_ids.append(cur_trace_id)

            # LOGIC FOR ADDING TO EXISTING DICTIOANRY
            # append to cur_dict what's in trace.anytracetype (orchestrationTrace) and put the whole thing in there

            if "orchestrationTrace" in cur_trace["trace"]:
                first_key = next(iter(cur_trace["trace"]["orchestrationTrace"]))
                cur_dict[first_key] = cur_trace["trace"]["orchestrationTrace"][
                    first_key
                ]

        if cur_dict:
            trace_steps.append(cur_dict)

        return trace_steps

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        trace = self._create_trace()

        # Invoke try block
        try:

            # Invoke tool and get processed response
            full_trace, processed_response, agent_start_time = self.invoke_agent()

            # if there is no response, then raise an error
            if not processed_response or not processed_response.get("agent_answer"):
                self._handle_error(
                    trace,
                    Exception("Failed to get or process agent response"),
                    "Agent Processing",
                )
                return None

            trace.update(
                metadata={
                    "Ground Truth": self.ground_truth,
                    str(self.eval_type + " Evaluation Model"): self.config[
                        "MODEL_ID_EVAL"
                    ],
                    "Chain of Thought Evaluation Model": self.config[
                        "MODEL_ID_EVAL_COT"
                    ],
                },
                output=processed_response["agent_answer"],
            )

            # Evaluation try block
            try:

                # Eliminate unneeded information for COT evaluation
                orc_trace_full = [
                    item["trace"]["orchestrationTrace"]
                    for item in full_trace
                    if "orchestrationTrace" in item["trace"]
                ]

                # Combine all the traces with the same trace ID
                trace_step_spans = self.combine_traces(full_trace)

                trimmed_orc_trace = [
                    item["rationale"]["text"]
                    for item in orc_trace_full
                    if "rationale" in item
                ]

                trace_steps = ""
                for i, item in enumerate(trimmed_orc_trace, 1):
                    trace_steps += f"Step {i}: {item}\n"

                agents_used = {self.agent_info["agentName"]}

                # Add collaborator agents if multi-agent in use
                if self.agent_info["agentType"] == "MULTI-AGENT":
                    agents_used = self._add_agent_collaborators(
                        agents_used, orc_trace_full
                    )

                # Chain of thought processes whole agent trace + agent info
                cot_eval_results, cot_system_prompt = cot_helper.evaluate_cot(
                    trace_steps,
                    processed_response["agent_answer"],
                    self.agent_info,
                    self.clients["bedrock_runtime"],
                    self.config["MODEL_ID_EVAL_COT"],
                )

                # Create an evaluation generation
                agent_generation = trace.generation(
                    name="Agent Generation Information",
                    input=[
                        {
                            "role": "system",
                            "content": self.agent_info["agentInstruction"],
                        },
                        {"role": "user", "content": self.question},
                    ],
                    model=self.agent_info["agentModel"],
                    model_parameters={"temperature": self.config["TEMPERATURE"]},
                    start_time=agent_start_time,
                    metadata=processed_response.get("agent_generation_metadata"),
                )

                agent_generation.end(
                    output=processed_response.get("agent_answer"),
                    usage_details={
                        "input": processed_response.get("input_tokens"),
                        "output": processed_response.get("output_tokens"),
                    },
                )

                # CHAIN OF THOUGHT EVALUATION SECTION START

                # Create generation based on CoT output
                cot_generation = trace.generation(
                    name="CoT Evaluation LLM-As-Judge Generation",
                    input=[
                        {"role": "system", "content": cot_system_prompt},
                        {"role": "user", "content": self.question},
                    ],
                    output=cot_eval_results,
                    metadata={
                        "agents_used": agents_used,
                        "model_used": self.config["MODEL_ID_EVAL_COT"],
                    },
                )

                for index, step in enumerate(trace_step_spans):

                    # Create trace step spans
                    subtrace_span = cot_generation.span(
                        name="Agent Trace Step {}".format(index + 1),
                        input=step.get("modelInvocationInput"),
                        output={
                            "Model Raw Response": step.get(
                                "modelInvocationOutput", {}
                            ).get("rawResponse"),
                            "Model Rationale": step.get("rationale"),
                        },
                        metadata={
                            "Model Output metadata": step.get(
                                "modelInvocationOutput", {}
                            ).get("metadata"),
                            "Observation": step.get("observation"),
                        },
                    )

                    subtrace_span.end()

                    # Prevents trace spans from getting sent out of order
                    time.sleep(1)

                cot_generation.end()

                # Send the scores of chain of thought evaluation
                for metric_name, value in cot_eval_results.items():
                    cot_generation.score(
                        name=str("COT_" + metric_name),
                        value=value["score"],
                        comment=value["explanation"],
                    )

                # CHAIN OF THOUGHT EVALUATION END

                # AGENT EVALAUATION RESULTS START

                # Prepare metadata and evaluate

                tool_usage_data = self.tool_usage_extractor.extract_from_trace(
                    full_trace
                )
                evaluation_metadata = {
                    "question": self.question,
                    "tool_usage": tool_usage_data,
                    "ground_truth": self.ground_truth,
                    "agent_response": processed_response.get("agent_answer"),
                    "evaluation_metadata": processed_response.get(
                        "agent_generation_metadata"
                    ),
                    **self.config,
                }

                evaluation_results = await self.evaluate_response(evaluation_metadata)

                # TODO: Make the logic better, stopgap solution to work with custom
                if self.eval_type != "CUSTOM":
                    for metric_name, metric_info in evaluation_results[
                        "metrics_scores"
                    ].items():
                        trace.score(
                            name=str(self.eval_type + "_" + metric_name),
                            value=metric_info.get("score"),
                            comment=metric_info.get("explanation"),
                        )

                # Update trace with final results
                return {
                    "question_id": self.question_id,
                    "question": self.question,
                    "ground_truth": self.ground_truth,
                    "agent_response": processed_response,
                    "evaluation_results": evaluation_results,
                    "trace_id": self.trace_id,
                }

            except Exception as e:
                self._handle_error(trace, e, "Evaluation")
                return None

        except Exception as e:
            self._handle_error(trace, e, "Agent Invocation")
            return None

        except KeyboardInterrupt as e:
            self._handle_error(trace, e, "Manually Stopped Evaluation Job")
            raise KeyboardInterrupt
