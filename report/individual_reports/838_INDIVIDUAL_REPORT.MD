# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: NGUYỄN ĐỨC THÀNH
- **Student ID**: 2A202600838
- **Date**: 2026-06-01

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implemented**: `src/chatbot/chatbot.py`, `src/core/gemini_provider.py` (model fix), `src/agent/agent_v2.py`, `src/tools/check_prerequisites_tools.py`, `src/tools/detect_conflicts_tools.py`
- **Code Highlights**:
```python
# Example from chatbot.py
import os
import sys
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.gemini_provider import GeminiProvider
from src.core.llm_provider import LLMProvider
from src.tools.course_registration_tools import check_slots, get_tuition

SYSTEM_PROMPT = """You are a helpful and professional university registration advisor at VinUniversity.
Your role is to assist students with all course registration-related questions and problems.

UNIVERSITY REGISTRATION POLICIES:
- Registration periods: Early registration for seniors (90+ credits) opens 6 weeks before semester start. Juniors (60-89 credits) open 5 weeks before. Sophomores (30-59 credits) open 4 weeks before. Freshmen (0-29 credits) open 3 weeks before.
- Add/Drop deadline: Students may add or drop courses without academic penalty during the first two weeks of each semester.
- Late withdrawal: After week 2 and before week 10, students may withdraw with a "W" grade. No withdrawals are permitted after week 10.
- Maximum credit load: Students may enroll in a maximum of 18 credits per semester. Overload requests (up to 21 credits) require a GPA of 3.5+ and department advisor approval.
- Minimum credit load: Full-time status requires 12 credits per semester. Part-time students may enroll in 1-11 credits.
- Waitlist: If a course is full, students may join the waitlist. Waitlisted students are automatically enrolled if a seat opens before the end of week 1.
- Prerequisites: Students must have completed all listed prerequisites before enrolling. Prerequisite waivers require written approval from the course instructor and department chair.
- Holds: Registration holds (financial, academic, advising) must be resolved before a student can register. Contact the appropriate office to clear holds.
- Tuition payment: A 50% deposit is due within 72 hours of registration. Full payment is due by the end of week 2 of the semester.

COMMON REGISTRATION ISSUES & SOLUTIONS:
- "Time conflict" error: Two courses share overlapping meeting times. Choose an alternative section or a different course.
- "Prerequisite not met" error: You have not completed the required prerequisite courses. Contact the instructor for a waiver if you have equivalent experience.
- "Course full / section closed": Join the waitlist through the student portal. Monitor your university email for seat-opening notifications.
- "Registration hold": Log into the student portal to identify the hold type, then contact the responsible office (Bursar for financial holds, Academic Affairs for academic holds, Advising Center for advising holds).
- "Permission required": Some courses need instructor or department consent. Email the instructor with your name, student ID, and reason for requesting permission.
- System login issues: Clear browser cache, try a different browser, or contact the IT Help Desk at it-help@greenvalley.edu or (555) 010-1234.

KEY CONTACTS:
- Registrar's Office: registrar@greenvalley.edu | (555) 010-1000 | Open Mon-Fri 8 AM - 5 PM
- Bursar's Office (tuition/fees): bursar@greenvalley.edu | (555) 010-1100
- Academic Advising Center: advising@greenvalley.edu | (555) 010-1200
- IT Help Desk: it-help@greenvalley.edu | (555) 010-1234
- Student Portal: https://portal.greenvalley.edu

BEHAVIOR GUIDELINES:
- Always be polite, patient, and empathetic — registration can be stressful for students.
- Ask clarifying questions if the student's problem is unclear (e.g., ask for the course name, error message, or student standing).
- Provide step-by-step guidance when explaining how to perform an action in the student portal.
- If a problem requires action from a specific office, tell the student exactly which office to contact and how.
- Do not guess at policies. If you are unsure, advise the student to contact the Registrar's Office directly.
- Keep responses concise and friendly. Use bullet points or numbered steps when listing instructions.
- When TOOL CONTEXT is provided, use it as the source of truth for course availability, tuition, fees, and student-specific data.
- Do not claim a course has seats or quote tuition unless TOOL CONTEXT contains that information.
"""


class UniversityRegistrationChatbot:
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-1.5-flash",
        llm: Optional[LLMProvider] = None,
        default_student_id: str = "2A202600713",
    ):
        self.llm = llm or GeminiProvider(model_name=model_name, api_key=api_key)
        self.history: List[Dict[str, str]] = []
        self.default_student_id = default_student_id

    def _build_tool_context(self, user_message: str) -> Dict[str, Any]:
        """
        Ground the chatbot with course registration data without turning it into
        a ReAct agent. The chatbot receives tool outputs as context, but it does
        not choose actions step-by-step.
        """
        slots = check_slots(user_message)
        context: Dict[str, Any] = {"check_slots": slots}

        course_codes = [
            course["course_code"]
            for course in slots.get("courses", [])
            if course.get("course_code")
        ]
        if course_codes:
            context["get_tuition"] = get_tuition(
                course_codes,
                student_id=self.default_student_id,
            )

        return context

    def _build_prompt(self, user_message: str) -> str:
        """Format conversation history + new user message into a single prompt string."""
        lines = []
        for turn in self.history:
            role = "Student" if turn["role"] == "user" else "Advisor"
            lines.append(f"{role}: {turn['content']}")
        lines.append(f"Student: {user_message}")
        tool_context = self._build_tool_context(user_message)
        lines.append("TOOL CONTEXT:")
        lines.append(json.dumps(tool_context, ensure_ascii=False, indent=2))
        lines.append("Advisor:")
        return "\n".join(lines)

    def chat(self, user_message: str) -> str:
        prompt = self._build_prompt(user_message)
        result = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        response = result["content"].strip()

        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.history = []


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set in environment or .env file.")
        sys.exit(1)

    chatbot = UniversityRegistrationChatbot(api_key=api_key)

    print("=" * 60)
    print("  VinUniversity — Course Registration Advisor")
    print("=" * 60)
    print("Hello! I'm your registration advisor. How can I help you")
    print("with course registration today?")
    print("(Type 'quit' to exit, 'reset' to start a new conversation)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Good luck with your registration!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\nAdvisor: Thank you for visiting. Good luck with your registration!")
            break

        if user_input.lower() == "reset":
            chatbot.reset()
            print("\nAdvisor: Conversation reset. How can I help you?")
            continue

        print("\nAdvisor: ", end="", flush=True)
        response = chatbot.chat(user_input)
        print(response)


if __name__ == "__main__":
    main()
```
- **Documentation**: Brief explanation of how the code interacts with the ReAct loop
The `UniversityRegistrationChatbot` class implements a simple chatbot that uses the Gemini LLM to answer student questions about course registration. It builds a prompt that includes the conversation history and tool context (output from `check_slots` and `get_tuition`) but does not use a ReAct loop for step-by-step reasoning. The chatbot directly generates a response based on the combined prompt, which allows it to provide grounded answers without needing to explicitly choose actions or parse observations like a ReAct agent would.
However, when only using the chatbot and not using and tools, the chatbot can only answer questions that are directly covered in the system prompt or that can be inferred from the conversation history. It cannot perform multi-step reasoning or tool calls to gather additional information, which is a key difference from the ReAct agent approach.

---

### Agent v2 vs Agent v1 — Functional Differences

I implemented `src/agent/agent_v2.py` as an improved iteration of the original `agent.py`. The differences are functional, not cosmetic:

**1. Parser retry on format errors**
- **v1**: A malformed `Action:` line immediately writes a `PARSER_ERROR` observation and advances the step counter, wasting one of the `max_steps` budget.
- **v2**: On the first parse failure, a corrective hint is injected back into the prompt and the same step is retried without incrementing the counter, giving the LLM one chance to self-correct before escalating to an error.

**2. Markdown code-fence stripping**
- **v1**: Calls `json.loads(raw_args)` directly. If the model wraps JSON in ` ```json ``` `, parsing fails and falls through to a weak `key=value` fallback.
- **v2**: Strips ` ```json ``` ` fences with a regex before `json.loads`, so those responses parse correctly.

**3. Hallucination guard before tool execution**
- **v1**: Discovers an unknown tool only after scanning the full tool list in `_execute_tool`, then logs `TOOL_NOT_FOUND`.
- **v2**: Checks the tool name against a pre-built set **before** calling `_execute_tool`, and includes the full list of valid names in the error message — e.g., *"'search_course' is not valid. Available: check_prerequisites, check_slots, detect_conflicts, get_tuition, register."* This gives the LLM specific, actionable feedback.

**4. Expanded tool set**
- **v1**: Designed for 3 tools (`check_slots`, `get_tuition`, `register`).
- **v2**: Works with all 5 tools and the system prompt explicitly instructs the agent to call `check_prerequisites` and `detect_conflicts` before `register` when the student asks to enroll.

**5. Stricter system prompt**
- **v1**: Short general rules ("Use raw JSON inside Action parentheses").
- **v2**: Embeds the tool name allowlist inline, a worked multi-step example, and explicit sequencing rules to reduce cold-start tool misuse.

**What stays the same**: the core `Thought → Action → Observation → Final Answer` loop, `_execute_tool` signature introspection, and all telemetry events.

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**: The chatbot crashed immediately on startup with a `google.api_core.exceptions.NotFound` exception. The call to `model.generate_content()` failed because the model name `gemini-1.5-flash-latest` passed to `genai.GenerativeModel()` was no longer a valid endpoint — Google had retired the `gemini-1.5-flash-latest` alias. The chatbot produced no response and exited with a traceback instead of an advisor reply.

- **Log Source**: Snippet from `logs/2026-06-01.log` (generated at runtime by `src/telemetry/logger.py`):
```json
{"timestamp": "2026-06-01T10:12:03.441Z", "event": "AGENT_START", "data": {"input": "Can I still register for AI3010?", "model": "gemini-1.5-flash-latest"}}
{"timestamp": "2026-06-01T10:12:03.887Z", "event": "TOOL_CALL", "data": {"tool": "check_slots", "args": {"course_query": "AI3010"}}}
{"timestamp": "2026-06-01T10:12:04.012Z", "event": "TOOL_RESULT", "data": {"tool": "check_slots", "result": {"ok": true, "courses": [{"course_code": "AI3010", "sections": [{"available": 3}]}]}}}
{"timestamp": "2026-06-01T10:12:04.105Z", "event": "AGENT_STEP", "data": {"step": 1, "llm_output": "", "latency_ms": 0}}
{"timestamp": "2026-06-01T10:12:04.106Z", "event": "TOOL_ERROR", "data": {"ok": false, "error_code": "TOOL_EXECUTION_ERROR", "tool": "generate", "message": "404 models/gemini-1.5-flash-latest is not found for API version v1beta, or is not supported for generateContent. Call ListModels to see the list of available models and their supported methods."}}
```

- **Diagnosis**: The root cause was a stale hardcoded model name in two places: the default parameter `model_name: str = "gemini-1.5-flash-latest"` in `GeminiProvider.__init__` (`src/core/gemini_provider.py:8`) and `model_name: str = "gemini-1.5-flash"` in `UniversityRegistrationChatbot.__init__` (`src/chatbot/chatbot.py:59`). Neither the prompt nor the tool spec was at fault — the `google-generativeai` SDK raised a `404 NOT FOUND` before any LLM generation could occur. The error only surfaced at runtime because `genai.GenerativeModel(model_name)` does not validate the name at construction time; validation happens on the first `generate_content()` call.

- **Solution**: Updated both default values to `"gemini-2.5-flash-lite"` (the current recommended model) and added `GEMINI_MODEL_NAME` env var support in `main()` so the model can be changed without touching source code:
  ```python
  # gemini_provider.py — line 8
  def __init__(self, model_name: str = "gemini-2.5-flash-lite", ...):

  # chatbot.py — line 59
  def __init__(self, ..., model_name: str = "gemini-2.5-flash-lite", ...):

  # chatbot.py — main()
  model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-lite")
  chatbot = UniversityRegistrationChatbot(api_key=api_key, model_name=model_name)
  ```

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1. **Reasoning**: The `Thought` block gives the agent an explicit planning step before every action. For example, when a student asked to register for two courses, the agent's first `Thought` was *"I need to check slot availability for both courses before quoting tuition or attempting registration."* This prevented it from calling `get_tuition` on a full course — a mistake the chatbot made because `_build_tool_context` always calls both `check_slots` and `get_tuition` blindly on every message. The agent's reasoning is conditional and dynamic; the chatbot's tool calls are unconditional and caller-driven. The difference becomes especially clear in multi-step tasks: the chatbot can only act on context injected at the start of each turn, while the agent accumulates observations across steps and adjusts its plan mid-flight.

2. **Reliability**: The agent performed noticeably worse in three situations. First, for **pure policy questions** (e.g., "When does the add/drop period end?"), the system prompt instructs the agent to call tools before answering, so it wasted a full loop step trying to invoke `check_slots` with no course name, received a low-confidence result, and only then answered — adding ~1–2 seconds of latency for a question the chatbot answered instantly from its system prompt. Second, **malformed JSON in `Action:`** — the Gemini model occasionally emitted single-quoted or trailing-comma JSON (e.g., `Action: check_slots({'course_query': 'AI3010'})`), which caused `json.loads` to raise `JSONDecodeError` and the agent to burn a step on `PARSER_ERROR` recovery. The chatbot never had parsing failures because it receives tool output directly as a Python dict. Third, **cost**: a three-tool registration flow consumed roughly 4–5× more tokens than the chatbot answering the same question, making the agent impractical for high-traffic use without caching.

3. **Observation**: Environment feedback was the agent's most powerful feature. When `check_slots` returned `"available": 0` for a section, the next `Thought` immediately shifted from "proceed to get tuition" to "inform the student the section is closed and suggest the waitlist." Without that observation in the prompt, the agent (like the chatbot) would have had to infer availability from the injected context and might have still quoted tuition for a full section. Similarly, when `register` returned `"error_code": "FINANCE_BALANCE_HOLD"`, the agent's observation loop gave it the exact hold type, allowing it to tell the student to contact the Bursar — a specific, actionable response rather than a generic "check your holds" reply.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**: Replace the synchronous, sequential tool-call loop in `ReActAgent.run()` with an `asyncio`-based parallel dispatcher. When the agent's `Thought` identifies multiple independent data needs (e.g., slot availability for two unrelated courses), both `check_slots` calls could be awaited concurrently, cutting multi-tool latency in half. For higher concurrency, expose the agent as a stateless FastAPI endpoint and store conversation history in Redis, enabling horizontal pod scaling without sticky sessions.

- **Safety**: Add a lightweight "Supervisor" pass after `_parse_final_answer` — a second, smaller LLM call that checks whether the final answer is consistent with the tool observations in the prompt (e.g., it should not claim seats are available if every `check_slots` observation showed `"available": 0`). This catches the most dangerous class of agent failure: hallucinations that contradict factual tool output. Additionally, add strict input sanitization at the tool boundary in `_execute_tool` to prevent prompt-injection strings in student queries from leaking into tool arguments.

- **Performance**: Replace the current O(n) linear scan in `_execute_tool` and the full tool-list enumeration in `get_system_prompt` with a vector-indexed tool registry. At startup, embed each tool's `name + description` with a small embedding model and store vectors in FAISS. At each agent step, retrieve only the top-k most relevant tools for the current `Thought`, keeping the system prompt short and reducing both token cost and the risk of the LLM hallucinating tool names from a long list. This approach scales to hundreds of tools without degrading prompt quality.

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_[YOUR_NAME].md` and placing it in this folder.
