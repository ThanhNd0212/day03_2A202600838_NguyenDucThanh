"""
Tool registry for the course registration assistant.

Individual tool logic lives in separate modules:
- check_slots_tools.py
- get_tuition_tools.py
- check_prerequisites_tools.py
- detect_conflicts_tools.py
- register_tools.py
"""
from typing import Any, Dict, List

from src.tools.check_slots_tools import check_slots, get_check_slots_tool
from src.tools.check_prerequisites_tools import check_prerequisites, get_check_prerequisites_tool
from src.tools.detect_conflicts_tools import detect_conflicts, get_detect_conflicts_tool
from src.tools.get_tuition_tools import get_get_tuition_tool, get_tuition
from src.tools.register_tools import get_register_tool, register


def get_course_registration_tools() -> List[Dict[str, Any]]:
    return [
        get_check_slots_tool(),
        get_get_tuition_tool(),
        get_check_prerequisites_tool(),
        get_detect_conflicts_tool(),
        get_register_tool(),
    ]


__all__ = [
    "check_slots",
    "get_tuition",
    "check_prerequisites",
    "detect_conflicts",
    "register",
    "get_course_registration_tools",
]
