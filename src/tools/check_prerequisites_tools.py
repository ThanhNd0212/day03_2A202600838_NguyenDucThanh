"""
Tool for checking whether a student meets course prerequisites.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "course_registration_mock.json"


@lru_cache(maxsize=1)
def _load_data() -> Dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_prerequisites(student_id: str, course_codes: List[str]) -> Dict[str, Any]:
    """
    Check whether a student has completed the prerequisites for each requested course.

    Args:
        student_id: The student's ID string.
        course_codes: List of course codes to check, e.g. ["AI3010", "DATA3020"].

    Returns:
        Dictionary with per-course eligibility and a top-level ok flag.
    """
    data = _load_data()

    student = next(
        (s for s in data["students"] if s["student_id"] == student_id), None
    )
    if not student:
        return {
            "ok": False,
            "student_id": student_id,
            "results": [],
            "errors": [f"Student {student_id} not found."],
        }

    completed = set(student.get("completed_courses", []))
    course_map: Dict[str, Dict[str, Any]] = {
        c["course_code"]: c for c in data["courses"]
    }

    results = []
    all_eligible = True
    errors = []

    for code in course_codes:
        course = course_map.get(code)
        if not course:
            errors.append(f"Course {code} not found.")
            results.append(
                {
                    "course_code": code,
                    "eligible": False,
                    "missing_prerequisites": [],
                    "error": f"Course {code} not found in catalog.",
                }
            )
            all_eligible = False
            continue

        required = course.get("prerequisites", [])
        missing = [req for req in required if req not in completed]
        eligible = len(missing) == 0
        if not eligible:
            all_eligible = False

        results.append(
            {
                "course_code": code,
                "title": course["title"],
                "required_prerequisites": required,
                "completed_prerequisites": [r for r in required if r in completed],
                "missing_prerequisites": missing,
                "eligible": eligible,
            }
        )

    return {
        "ok": all_eligible and not errors,
        "student_id": student_id,
        "student_name": student["full_name"],
        "results": results,
        "errors": errors,
    }


def get_check_prerequisites_tool() -> Dict[str, Any]:
    return {
        "name": "check_prerequisites",
        "description": (
            "Check whether a student has completed the prerequisites for one or more courses. "
            "Input JSON: {\"student_id\": \"2A202600713\", \"course_codes\": [\"AI3010\", \"DATA3020\"]}. "
            "Returns per-course eligibility and a list of any missing prerequisites."
        ),
        "function": check_prerequisites,
    }
