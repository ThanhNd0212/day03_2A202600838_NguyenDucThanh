"""
ReAct Agent v2 — improved over v1 with:
  1. Stricter JSON-only Action format to eliminate markdown/single-quote parse errors.
  2. One automatic retry when a PARSER_ERROR occurs, injecting a corrective hint.
  3. Explicit tool-name allowlist check before execution to surface hallucinations early.
  4. Richer system prompt with a worked example to reduce cold-start tool misuse.
"""
import inspect
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

_MAX_PARSER_RETRIES = 1


class ReActAgentV2:
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 7):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self._tool_names = {t["name"] for t in tools}

    def get_system_prompt(self) -> str:
        tool_lines = "\n".join(
            f"  - {t['name']}: {t['description']}" for t in self.tools
        )
        tool_name_list = ", ".join(sorted(self._tool_names))
        return f"""You are a VinUniversity course registration assistant (Agent v2).

AVAILABLE TOOLS (use ONLY these names: {tool_name_list}):
{tool_lines}

STRICT OUTPUT FORMAT — follow this exactly, no exceptions:
  Thought: one sentence explaining what you need to do next.
  Action: tool_name({{"key": "value"}})

Rules:
  - Action arguments MUST be valid JSON inside the parentheses. No single quotes, no trailing commas.
  - Never invent tool names. Only call tools from the list above.
  - Call check_slots before get_tuition or register.
  - Call check_prerequisites and detect_conflicts before register when the student asks to enroll.
  - After all needed observations, write:
      Final Answer: <your complete answer to the student>

EXAMPLE (do not copy verbatim — adapt to the actual question):
  Thought: I need to check seat availability for AI3010.
  Action: check_slots({{"course_query": ["AI3010"]}})
  Observation: {{ ... }}
  Thought: Seats are available; now I need the tuition.
  Action: get_tuition({{"course_code": ["AI3010"], "student_id": "2A202600713"}})
  Observation: {{ ... }}
  Final Answer: AI3010 has 4 open seats. Estimated tuition is 9,750,000 VND.
"""

    def run(self, user_input: str) -> str:
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name, "version": "v2"})

        prompt = f"Question: {user_input}"
        steps = 0
        parser_retries = 0

        while steps < self.max_steps:
            result = self.llm.generate(prompt, system_prompt=self.get_system_prompt())
            content = result.get("content", "")

            logger.log_event(
                "AGENT_STEP",
                {"step": steps + 1, "llm_output": content, "latency_ms": result.get("latency_ms", 0)},
            )
            if "usage" in result:
                tracker.track_request(
                    provider=result.get("provider", "unknown"),
                    model=self.llm.model_name,
                    usage=result.get("usage", {}),
                    latency_ms=result.get("latency_ms", 0),
                )

            final = self._parse_final_answer(content)
            if final:
                logger.log_event("AGENT_END", {"steps": steps + 1, "status": "final_answer", "version": "v2"})
                return final

            action = self._parse_action(content)

            # v2 improvement: retry once with a corrective hint on parse failure
            if action is None:
                logger.log_event("AGENT_PARSE_ERROR", {"step": steps + 1, "output": content})
                if parser_retries < _MAX_PARSER_RETRIES:
                    parser_retries += 1
                    hint = (
                        "\nObservation: PARSE ERROR — your last Action line could not be parsed. "
                        "Rewrite it using ONLY double-quoted JSON, for example:\n"
                        '  Action: check_slots({"course_query": ["AI3010"]})\n'
                        "Do not use markdown code fences or single quotes."
                    )
                    prompt = f"{prompt}\n{content}{hint}"
                    continue
                observation = json.dumps(
                    {
                        "ok": False,
                        "error_code": "PARSER_ERROR",
                        "message": "Action format invalid after retry. Use Action: tool_name({\"arg\": \"value\"}).",
                    }
                )
            else:
                tool_name, args = action

                # v2 improvement: explicit hallucination check before execution
                if tool_name not in self._tool_names:
                    observation = json.dumps(
                        {
                            "ok": False,
                            "error_code": "TOOL_NOT_FOUND",
                            "message": f"'{tool_name}' is not a valid tool. Available: {', '.join(sorted(self._tool_names))}.",
                        }
                    )
                    logger.log_event("TOOL_ERROR", {"tool": tool_name, "error_code": "TOOL_NOT_FOUND"})
                else:
                    observation = self._execute_tool(tool_name, args)

            prompt = f"{prompt}\n{content}\nObservation: {observation}"
            steps += 1
            parser_retries = 0

        logger.log_event("AGENT_END", {"steps": steps, "status": "max_steps_exceeded", "version": "v2"})
        return "I could not complete the registration task within the allowed reasoning steps."

    def _parse_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r"Final Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_action(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        match = re.search(
            r"Action\s*:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$",
            text.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return None

        tool_name = match.group(1)
        raw_args = match.group(2).strip()
        if not raw_args:
            return tool_name, {}

        # Strip markdown code fences if the model wrapped the JSON (v2 robustness)
        raw_args = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_args.strip())

        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed = self._parse_key_value_args(raw_args)

        if not isinstance(parsed, dict):
            parsed = {"value": parsed}

        return tool_name, parsed

    def _parse_key_value_args(self, raw: str) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for item in raw.split(","):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            args[key.strip()] = value.strip().strip("\"'")
        return args

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        for tool in self.tools:
            if tool["name"] == tool_name:
                try:
                    fn = tool["function"]
                    sig = inspect.signature(fn)
                    filtered = {k: v for k, v in args.items() if k in sig.parameters}
                    logger.log_event("TOOL_CALL", {"tool": tool_name, "args": filtered})
                    result = fn(**filtered)
                    logger.log_event("TOOL_RESULT", {"tool": tool_name, "result": result})
                    return json.dumps(result, ensure_ascii=False)
                except Exception as exc:
                    error = {
                        "ok": False,
                        "error_code": "TOOL_EXECUTION_ERROR",
                        "tool": tool_name,
                        "message": str(exc),
                    }
                    logger.log_event("TOOL_ERROR", error)
                    return json.dumps(error, ensure_ascii=False)

        error = {"ok": False, "error_code": "TOOL_NOT_FOUND", "message": f"Tool {tool_name} not found."}
        logger.log_event("TOOL_ERROR", error)
        return json.dumps(error, ensure_ascii=False)
