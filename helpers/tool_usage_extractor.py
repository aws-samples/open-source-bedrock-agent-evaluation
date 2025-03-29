class ToolUsageExtractor:
    """
    A class to extract tool usage information from AWS Bedrock Agent trace objects.
    Identifies agents involved in the conversation and their tool calls with parameters.
    """

    def __init__(self):
        """Initialize the ToolUsageExtractor"""
        pass

    def extract_from_trace(self, trace_events):
        """
        Extract tool usage information from a Bedrock Agent trace object.

        Args:
            trace_data (dict): The trace object from Bedrock's invokeAgent API response

        Returns:
            dict: A dictionary containing agents and their tool usage
        """
        result = {"agents": {}, "tool_calls": []}

        for trace_event in trace_events:
            try:
                if "agentId" in trace_event:
                    agent_id = trace_event["agentId"]
                    agent_alias_id = trace_event.get("agentAliasId", "")
                    agent_name = trace_event.get(
                        "collaboratorName", f"Agent-{agent_id}"
                    )

                    if agent_id not in result["agents"]:
                        result["agents"][agent_id] = {
                            "agent_id": agent_id,
                            "agent_alias_id": agent_alias_id,
                            "agent_name": agent_name,
                            "tool_calls": [],
                        }

                # Extract caller chain if available (for multi-agent scenarios)
                if "callerChain" in trace_event:
                    caller_chain = trace_event["callerChain"]
                    for caller in caller_chain:
                        if "agentAliasArn" in caller:
                            arn_parts = caller["agentAliasArn"].split("/")
                            if len(arn_parts) >= 2:
                                caller_agent_id = arn_parts[-2]
                                if caller_agent_id not in result["agents"]:
                                    result["agents"][caller_agent_id] = {
                                        "agent_id": caller_agent_id,
                                        "agent_alias_id": (
                                            arn_parts[-1] if len(arn_parts) > 2 else ""
                                        ),
                                        "agent_name": f"Agent-{caller_agent_id}",
                                        "tool_calls": [],
                                    }

                if (
                    "trace" in trace_event
                    and "orchestrationTrace" in trace_event["trace"]
                ):
                    orchestration_trace = trace_event["trace"]["orchestrationTrace"]

                    if "invocationInput" in orchestration_trace:
                        invocation_input = orchestration_trace["invocationInput"]

                        if "actionGroupInvocationInput" in invocation_input:
                            action_group_input = invocation_input[
                                "actionGroupInvocationInput"
                            ]

                            if "function" in action_group_input:
                                tool_name = action_group_input.get("function")
                                parameters = action_group_input.get("parameters", {})
                            elif "apiPath" in action_group_input:
                                # open-api API-based action group
                                tool_name = action_group_input.get("apiPath")
                                parameters = action_group_input.get("requestBody", {})
                            else:
                                continue

                            tool_call = {
                                "agent_id": trace_event.get("agentId", "unknown"),
                                "tool_name": tool_name,
                                "parameters": parameters,
                                "timestamp": trace_event.get("timestamp", ""),
                            }

                            result["tool_calls"].append(tool_call)

                            if tool_call["agent_id"] in result["agents"]:
                                result["agents"][tool_call["agent_id"]][
                                    "tool_calls"
                                ].append(
                                    {
                                        "tool_name": tool_call["tool_name"],
                                        "parameters": tool_call["parameters"],
                                        "timestamp": tool_call["timestamp"],
                                    }
                                )
            except Exception as e:
                print(f"Error processing trace event for tool extraction: {str(e)}")
                print(f"Trace event data: {trace_event}")

        return result

    def extract_from_response(self, invoke_agent_response):
        """
        Extract tool usage from the full invoke_agent response object.

        Args:
            invoke_agent_response (dict): The full response from invokeAgent API

        Returns:
            dict: A dictionary containing agents and their tool usage
        """
        trace_events = []

        # Extract trace events from the completion stream
        if "completion" in invoke_agent_response:
            for event in invoke_agent_response["completion"]:
                if "trace" in event:
                    trace_events.append(event["trace"])

        return self.extract_from_trace(trace_events)
