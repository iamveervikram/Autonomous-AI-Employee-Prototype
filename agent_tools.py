import difflib
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.tools import tool
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field

from mock_system.database import (
    get_all_employees,
    get_employee_by_id,
    init_db,
    log_audit_event,
    update_vault_status,
)

load_dotenv()

PORTAL_BASE_URL = os.getenv("PORTAL_BASE_URL", "http://localhost:8000").rstrip("/")
HEADLESS_BROWSER = os.getenv("HEADLESS_BROWSER", "false").lower() in ("true", "1", "yes")

ACTIVE_SSO_SESSIONS: Dict[str, bool] = {}


def reset_sso_session_locks() -> None:
    ACTIVE_SSO_SESSIONS.clear()


class SearchEmployeeInput(BaseModel):
    name: str = Field(..., description="Full or partial name of the employee to look up in the HR database.")


class EmployeeIdInput(BaseModel):
    employee_id: str = Field(..., description="The unique Employee ID (e.g., 'EMP-001' or 'EMP-002').")


class ReasoningPlanInput(BaseModel):
    intended_outcome: str = Field(..., description="Clear statement of the user's intended business outcome.")
    what_needs_to_be_done: str = Field(
        ..., description="Summary of the procedures and policy checks required to achieve the outcome."
    )
    planned_steps: List[str] = Field(
        ..., description="Ordered list of concrete actions and tools the agent plans to execute."
    )


class HumanClarificationInput(BaseModel):
    question: str = Field(
        ...,
        description="Question or confirmation prompt for the human operator when an employee name is ambiguous.",
    )
    candidate_id: str = Field(default="", description="Candidate employee ID found in the database.")
    candidate_name: str = Field(default="", description="Candidate employee name found in the database.")


def _normalize_tokens(name: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-zA-Z0-9]+", name.lower()) if tok]


@tool("record_execution_plan", args_schema=ReasoningPlanInput)
def record_execution_plan(
    intended_outcome: str,
    what_needs_to_be_done: str,
    planned_steps: List[str],
) -> str:
    """Record the intended outcome, required work, and ordered action plan before executing tools."""
    steps_formatted = "\n".join([f"  {idx + 1}. {step}" for idx, step in enumerate(planned_steps)])
    log_audit_event("PLANNER", "EXECUTION_PLAN_CREATED", f"Outcome: {intended_outcome}")
    return (
        f"PLAN_RECORDED:\n"
        f"- Intended Outcome: {intended_outcome}\n"
        f"- Required Work: {what_needs_to_be_done}\n"
        f"- Action Plan:\n{steps_formatted}"
    )


@tool("search_employee_db", args_schema=SearchEmployeeInput)
def search_employee_db(name: str) -> str:
    """Search the HR database by employee name. Returns exact match or fuzzy candidates."""
    try:
        init_db(reset=False)
        query_name = name.strip()
        if not query_name:
            return "ERROR: Employee name query cannot be empty."

        all_employees = get_all_employees()
        if not all_employees:
            return "ERROR: The HR database is empty."

        exact_matches = [
            emp for emp in all_employees if emp["name"].strip().lower() == query_name.lower()
        ]
        if len(exact_matches) == 1:
            emp = exact_matches[0]
            log_audit_event(emp["id"], "DB_SEARCH_EXACT_MATCH", f"Exact match found for query '{query_name}'.")
            return (
                f"EXACT_MATCH_FOUND: ID='{emp['id']}', Name='{emp['name']}', "
                f"Email='{emp['email']}', Role='{emp['role']}', Department='{emp['department']}', "
                f"AccountStatus='{emp['status']}', VaultStatus='{emp['vault_status']}'."
            )

        query_tokens = _normalize_tokens(query_name)
        candidates = []

        for emp in all_employees:
            emp_name = emp["name"]
            emp_tokens = _normalize_tokens(emp_name)
            shared_tokens = set(query_tokens).intersection(set(emp_tokens))

            initial_matches = 0
            for qt in query_tokens:
                for et in emp_tokens:
                    if qt != et and (
                        (len(et) == 1 and qt.startswith(et))
                        or (len(qt) == 1 and et.startswith(qt))
                    ):
                        initial_matches += 1

            seq_ratio = difflib.SequenceMatcher(None, query_name.lower(), emp_name.lower()).ratio()

            if len(shared_tokens) >= 1 or seq_ratio >= 0.6:
                score = (len(shared_tokens) * 0.4) + (initial_matches * 0.25) + (seq_ratio * 0.35)
                candidates.append((score, emp))

        candidates.sort(key=lambda item: item[0], reverse=True)

        if candidates:
            formatted_candidates = [
                f"[ID='{emp['id']}', Name='{emp['name']}', Email='{emp['email']}', "
                f"Department='{emp['department']}', AccountStatus='{emp['status']}', VaultStatus='{emp['vault_status']}']"
                for _, emp in candidates
            ]
            candidates_str = "; ".join(formatted_candidates)
            log_audit_event(
                "SEARCH",
                "DB_SEARCH_AMBIGUOUS",
                f"No exact match for '{query_name}'. Candidates: {candidates_str}",
            )
            return (
                f"NO_EXACT_MATCH: No employee found with the exact name '{query_name}'. "
                f"However, found {len(candidates)} similar candidate(s) in the database: {candidates_str}. "
                f"POLICY REQUIREMENT: Because the name does not match 100% exactly, you MUST call "
                f"`ask_human_clarification` to confirm with the human operator whether to proceed with the candidate before taking any deactivation action."
            )

        available_names = ", ".join([f"{e['name']} ({e['id']})" for e in all_employees])
        return (
            f"NOT_FOUND: No employee matching '{query_name}' was found in the HR database. "
            f"Currently registered employees are: {available_names}. Try variations of the name or ask the human operator."
        )

    except Exception as exc:
        return f"ERROR querying database: {str(exc)}"


@tool("ask_human_clarification", args_schema=HumanClarificationInput)
def ask_human_clarification(question: str, candidate_id: str = "", candidate_name: str = "") -> str:
    """Pause and ask the human operator to confirm an ambiguous employee match."""
    return f"Human operator approved proceeding with candidate {candidate_name} ({candidate_id})."


@tool("terminate_active_sso_sessions", args_schema=EmployeeIdInput)
def terminate_active_sso_sessions(employee_id: str) -> str:
    """Force-terminate active SSO & VPN sessions when account deactivation is blocked by a session lock."""
    emp_id = employee_id.strip().upper()
    emp = get_employee_by_id(emp_id)
    if not emp:
        return f"SSO_ERROR: Employee ID '{emp_id}' not found."

    ACTIVE_SSO_SESSIONS[emp_id] = False
    log_audit_event(
        emp_id,
        "SSO_SESSIONS_REVOKED",
        f"Force-terminated 2 active SSO/VPN sessions for {emp['name']} to clear deactivation lock.",
    )
    return (
        f"RECOVERY_ACTION_SUCCESS: Force-terminated 2 active SSO/VPN sessions (Okta token & WireGuard tunnel) "
        f"for {emp['name']} ({emp_id}). The session lock is now cleared. You may now retry `deactivate_user_via_browser`."
    )


@tool("deactivate_user_via_browser", args_schema=EmployeeIdInput)
def deactivate_user_via_browser(employee_id: str) -> str:
    """Open the internal HR portal via Playwright and click the 'Deactivate Account' button."""
    emp_id = employee_id.strip().upper()
    target_url = f"{PORTAL_BASE_URL}/employee/{emp_id}"

    if ACTIVE_SSO_SESSIONS.get(emp_id, True):
        log_audit_event(
            emp_id,
            "DEACTIVATION_BLOCKED_SSO_LOCK",
            "Browser deactivation blocked: Employee has 2 active SSO/VPN sessions open.",
        )
        return (
            f"SESSION_LOCK_ERROR (HTTP 409 Conflict): Attempted to deactivate {emp_id} on {target_url}, "
            f"but the HR Portal rejected the request because {emp_id} currently has 2 active SSO/VPN sessions open. "
            f"RECOVERY HINT: Call `terminate_active_sso_sessions(employee_id='{emp_id}')` to revoke active sessions, "
            f"and then retry `deactivate_user_via_browser(employee_id='{emp_id}')`."
        )

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=HEADLESS_BROWSER,
                slow_mo=900 if not HEADLESS_BROWSER else 0,
            )
            page = browser.new_page()

            page.goto(PORTAL_BASE_URL, wait_until="domcontentloaded", timeout=10000)
            response = page.goto(target_url, wait_until="domcontentloaded", timeout=10000)
            if response and response.status == 404:
                browser.close()
                return (
                    f"BROWSER_ERROR: HR Portal returned HTTP 404 for {target_url}. "
                    f"Employee ID '{emp_id}' does not exist. Please check the employee ID."
                )

            emp_name_locator = page.locator("[data-test-id='employee-name']")
            emp_name = emp_name_locator.inner_text().strip() if emp_name_locator.count() > 0 else emp_id

            status_badge = page.locator("[data-test-id='account-status-badge']")
            initial_status = status_badge.inner_text().strip().lower() if status_badge.count() > 0 else "unknown"

            if initial_status == "inactive":
                browser.close()
                return (
                    f"BROWSER_OBSERVATION: Navigated to {target_url} for {emp_name} ({emp_id}). "
                    f"The account status badge is already 'inactive' and the Deactivate button is disabled."
                )

            deactivate_btn = page.locator("[data-test-id='deactivate-btn']")
            if deactivate_btn.count() == 0:
                browser.close()
                return (
                    f"BROWSER_ERROR: Could not find '[data-test-id=deactivate-btn]' on {target_url}. "
                    f"Current status badge reads: '{initial_status}'."
                )

            deactivate_btn.click()
            page.wait_for_load_state("domcontentloaded", timeout=10000)

            updated_badge = page.locator("[data-test-id='account-status-badge']")
            updated_status = updated_badge.inner_text().strip().lower() if updated_badge.count() > 0 else "unknown"

            flash_locator = page.locator("[data-test-id='flash-message']")
            flash_text = flash_locator.inner_text().strip() if flash_locator.count() > 0 else "No banner message"

            if not HEADLESS_BROWSER:
                page.wait_for_timeout(1200)

            browser.close()

            return (
                f"BROWSER_ACTION_COMPLETED: Navigated to {target_url}, clicked '#deactivate-btn' for "
                f"{emp_name} ({emp_id}). Portal flash banner: '{flash_text}'. "
                f"Observed UI status badge changed from '{initial_status}' -> '{updated_status}'."
            )

    except PlaywrightTimeoutError:
        return (
            f"BROWSER_TIMEOUT_ERROR: Timed out attempting to reach {target_url}. "
            f"Ensure the FastAPI mock HR portal server is running at {PORTAL_BASE_URL}."
        )
    except Exception as exc:
        return (
            f"BROWSER_EXECUTION_ERROR: Failed to operate browser on {target_url}. "
            f"Details: {str(exc)}. Ensure the FastAPI server is running on {PORTAL_BASE_URL}."
        )


@tool("delete_user_data_vault", args_schema=EmployeeIdInput)
def delete_user_data_vault(employee_id: str) -> str:
    """Permanently purge the employee's personal data vault (requires Human-in-the-Loop approval)."""
    emp_id = employee_id.strip().upper()
    emp = get_employee_by_id(emp_id)
    if not emp:
        return f"VAULT_ERROR: Employee ID '{emp_id}' not found in database."

    if emp["vault_status"] == "purged":
        return f"VAULT_OBSERVATION: Data vault for {emp['name']} ({emp_id}) was already purged previously."

    success = update_vault_status(emp_id, "purged")
    if not success:
        return f"VAULT_ERROR: Failed to update vault status for '{emp_id}'."

    return (
        f"VAULT_DELETED_SUCCESS: Permanently purged personal data vault, API tokens, and encrypted "
        f"backups for {emp['name']} ({emp_id}). Vault status is now 'purged'."
    )


@tool("verify_deactivation", args_schema=EmployeeIdInput)
def verify_deactivation(employee_id: str) -> str:
    """Cross-check both the live HR Web Portal DOM and the SQLite database to verify offboarding completion."""
    emp_id = employee_id.strip().upper()
    db_emp = get_employee_by_id(emp_id)
    if not db_emp:
        return f"VERIFICATION_FAILED: Employee ID '{emp_id}' does not exist in the SQLite database."

    ui_status = "unreachable"
    ui_vault_status = "unreachable"
    target_url = f"{PORTAL_BASE_URL}/employee/{emp_id}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            response = page.goto(target_url, wait_until="domcontentloaded", timeout=8000)
            if response and response.status == 200:
                badge = page.locator("[data-test-id='account-status-badge']")
                if badge.count() > 0:
                    ui_status = badge.inner_text().strip().lower()
                vault_el = page.locator("[data-test-id='vault-status-value']")
                if vault_el.count() > 0:
                    ui_vault_status = vault_el.inner_text().strip().lower()
            browser.close()
    except Exception as exc:
        ui_status = f"error ({str(exc)})"

    is_deactivated = (db_emp["status"] == "inactive") and (ui_status == "inactive")
    log_audit_event(
        emp_id,
        "OUTCOME_VERIFICATION",
        f"Verified offboarding: db_status={db_emp['status']}, ui_status={ui_status}, vault={db_emp['vault_status']}",
    )

    verification_verdict = "VERIFIED_SUCCESS" if is_deactivated else "VERIFICATION_MISMATCH"
    return (
        f"{verification_verdict}: Offboarding verification report for {db_emp['name']} ({emp_id}):\n"
        f" - Database Account Status: '{db_emp['status']}'\n"
        f" - Web Portal UI Badge Status ({target_url}): '{ui_status}'\n"
        f" - Data Vault Status: '{db_emp['vault_status']}' (UI shows: '{ui_vault_status}')\n"
        f" - Active SSO Sessions Remaining: 0\n"
        f" - Last Updated Timestamp: '{db_emp['updated_at']}'"
    )


ALL_TOOLS = [
    record_execution_plan,
    search_employee_db,
    ask_human_clarification,
    deactivate_user_via_browser,
    terminate_active_sso_sessions,
    delete_user_data_vault,
    verify_deactivation,
]

HITL_TOOL_NAMES = {"ask_human_clarification", "delete_user_data_vault"}
