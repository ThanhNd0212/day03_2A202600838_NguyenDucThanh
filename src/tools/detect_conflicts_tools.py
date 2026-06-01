"""
Tool for detecting schedule conflicts between requested and existing registrations.
"""
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "course_registration_mock.json"


@lru_cache(maxsize=1)
def _load_data() -> Dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _time_to_minutes(value: str) -> int:
    parsed = datetime.strptime(value, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def _sections_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    for slot_a in a["schedule"]:
        for slot_b in b["schedule"]:
            if slot_a["day"] != slot_b["day"]:
                continue
            a_start = _time_to_minutes(slot_a["start"])
            a_end = _time_to_minutes(slot_a["end"])
            b_start = _time_to_minutes(slot_b["start"])
            b_end = _time_to_minutes(slot_b["end"])
            if a_start < b_end and b_start < a_end:
                return True
    return False


def _find_section(
    section_id: str, data: Dict[str, Any]
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for course in data["courses"]:
        for section in course["sections"]:
            if section["section_id"] == section_id:
                return course, section
    return None


def detect_conflicts(student_id: str, section_ids: List[str]) -> Dict[str, Any]:
    """
    Detect schedule conflicts between requested sections and the student's
    existing registrations, and also among the requested sections themselves.

    Args:
        student_id: The student's ID string.
        section_ids: List of section IDs the student wants to add.

    Returns:
        Dictionary listing any conflicting section pairs.
    """
    data = _load_data()

    student = next(
        (s for s in data["students"] if s["student_id"] == student_id), None
    )
    if not student:
        return {
            "ok": False,
            "student_id": student_id,
            "conflicts": [],
            "errors": [f"Student {student_id} not found."],
        }

    errors = []
    requested: List[Tuple[str, Dict[str, Any]]] = []
    for sid in section_ids:
        found = _find_section(sid, data)
        if not found:
            errors.append(f"Section {sid} not found.")
        else:
            requested.append((sid, found[1]))

    existing: List[Tuple[str, Dict[str, Any]]] = []
    for sid in student.get("current_registrations", []):
        found = _find_section(sid, data)
        if found:
            existing.append((sid, found[1]))

    conflicts = []

    # Check requested vs existing
    for req_id, req_section in requested:
        for ex_id, ex_section in existing:
            if _sections_overlap(req_section, ex_section):
                conflicts.append(
                    {
                        "section_a": req_id,
                        "section_b": ex_id,
                        "type": "new_vs_existing",
                        "message": (
                            f"{req_id} (requested) conflicts with {ex_id} (already registered)."
                        ),
                    }
                )

    # Check requested sections among themselves
    for i, (id_a, sec_a) in enumerate(requested):
        for id_b, sec_b in requested[i + 1:]:
            if _sections_overlap(sec_a, sec_b):
                conflicts.append(
                    {
                        "section_a": id_a,
                        "section_b": id_b,
                        "type": "within_request",
                        "message": (
                            f"{id_a} and {id_b} (both requested) overlap in schedule."
                        ),
                    }
                )

    return {
        "ok": len(conflicts) == 0 and not errors,
        "student_id": student_id,
        "student_name": student["full_name"],
        "conflicts": conflicts,
        "conflict_free": len(conflicts) == 0,
        "errors": errors,
    }


def get_detect_conflicts_tool() -> Dict[str, Any]:
    return {
        "name": "detect_conflicts",
        "description": (
            "Detect schedule conflicts between the student's requested sections and their existing "
            "registrations, and also among the requested sections themselves. "
            "Input JSON: {\"student_id\": \"2A202600713\", \"section_ids\": [\"AI3010-01\", \"DATA3020-02\"]}. "
            "Returns a list of conflicting section pairs and a conflict_free boolean."
        ),
        "function": detect_conflicts,
    }
