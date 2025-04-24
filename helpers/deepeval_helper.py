from typing import List, Dict, Any
from deepeval.test_case import ToolCall


class DeepevalHelper:
    """Helper class for converting tool calls to DeepEval-compatible format"""

    @staticmethod
    def convert_to_deepeval_tool_calls(
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert a list of tool calls from the sample format to DeepEval tool call format.

        Parameters:
            tool_calls: List of tool calls in the format:
                [
                    {
                        "tool_name": "tool-name",
                        "parameters": {
                            "name1": "value1"
                        }
                    },
                    ...
                ]

        Returns:
            List of DeepEval tool call objects:
        """
        deepeval_tool_calls = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name", "")
            parameters = tool_call.get("parameters", [])

            # Create DEEPEVAL ToolCall object
            deepeval_tool_call = ToolCall(name=tool_name)
            deepeval_tool_calls.append(deepeval_tool_call)

        return deepeval_tool_calls

    @staticmethod
    def convert_bedrock_format_to_deepeval_tool_calls(
        bedrock_tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert tool calls from Bedrock format to DeepEval format.

        Parameters:
            bedrock_tool_calls: List of tool calls in Bedrock format:
                [
                    {
                        "name": "tool-name",
                        "parameters": [
                            {
                                "name": "param1",
                                "type": "type",
                                "value": "value1"
                            }
                        ]
                    },
                    ...
                ]

        Returns:
            List of DeepEval tool call objects
        """
        deepeval_tool_calls = []

        for call in bedrock_tool_calls:
            parameters = {}
            for param in call.get("parameters", []):
                parameters[param.get("name")] = param.get("value")

            tool_call = ToolCall(name=call.get("tool_name", ""))
            deepeval_tool_calls.append(tool_call)

        return deepeval_tool_calls
