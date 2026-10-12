import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from agent_graph import build_agent_graph
from agent_tools import HITL_TOOL_NAMES, reset_sso_session_locks
from mock_system.database import (
    get_all_employees,
    get_employee_by_id,
    get_recent_audit_logs,
    init_db,
    update_employee_status,
    update_vault_status,
)

app = FastAPI(title="CentrAlign AI — Autonomous AI Employee")

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

AGENT_GRAPH = build_agent_graph()
AGENT_SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSIONS_LOCK = threading.Lock()


class StartAgentRequest(BaseModel):
    task: str
    reset_db: bool = False


class HitlDecisionRequest(BaseModel):
    session_id: str
    decision: str


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _append_step(
    session_id: str,
    step_type: str,
    phase_badge: str,
    title: str,
    detail: str,
) -> None:
    with SESSIONS_LOCK:
        if session_id in AGENT_SESSIONS:
            AGENT_SESSIONS[session_id]["steps"].append(
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "type": step_type,
                    "phase": phase_badge,
                    "title": title,
                    "detail": detail,
                }
            )


def _update_criteria(session_id: str, *criteria_keys: str) -> None:
    with SESSIONS_LOCK:
        if session_id in AGENT_SESSIONS:
            for key in criteria_keys:
                AGENT_SESSIONS[session_id]["criteria"][key] = True


def _execute_agent_worker(session_id: str, initial_input: Any) -> None:
    config = {"configurable": {"thread_id": session_id}}
    current_input = initial_input

    try:
        while True:
            for event in AGENT_GRAPH.stream(current_input, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if not isinstance(node_output, dict):
                        continue
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            text = _extract_text(msg.content).strip()
                            if text:
                                _update_criteria(session_id, "evidence_summary", "state_context")
                                _append_step(
                                    session_id,
                                    "ai_message",
                                    "Evidence & Summary",
                                    "Final Outcome & Verification Summary",
                                    text,
                                )
                            if getattr(msg, "tool_calls", None):
                                for tc in msg.tool_calls:
                                    t_name = tc["name"]
                                    t_args = tc["args"]
                                    _update_criteria(session_id, "selecting_tools", "executing_actions", "state_context")

                                    if t_name == "record_execution_plan":
                                        _update_criteria(
                                            session_id,
                                            "understanding_outcome",
                                            "determining_work",
                                            "planning_actions",
                                        )
                                        planned_list = t_args.get("planned_steps", [])
                                        steps_str = "\n".join(
                                            [f"{i+1}. {s}" for i, s in enumerate(planned_list)]
                                        )
                                        plan_detail = (
                                            f"Intended Outcome : {t_args.get('intended_outcome', '')}\n"
                                            f"Required Work    : {t_args.get('what_needs_to_be_done', '')}\n"
                                            f"Planned Actions  :\n{steps_str}"
                                        )
                                        _append_step(
                                            session_id,
                                            "plan",
                                            "Understand & Plan",
                                            "Formulated Execution Plan & Selected Tools",
                                            plan_detail,
                                        )
                                    elif t_name == "terminate_active_sso_sessions":
                                        _update_criteria(session_id, "recovering_failures", "determining_next")
                                        _append_step(
                                            session_id,
                                            "recovery",
                                            "Recovering from Failure",
                                            f"Self-Healing Action: {t_name}",
                                            f"Revoking active SSO/VPN locks for employee_id='{t_args.get('employee_id')}' so browser deactivation can succeed.",
                                        )
                                    elif t_name == "verify_deactivation":
                                        _update_criteria(session_id, "verifying_outcome", "determining_next")
                                        _append_step(
                                            session_id,
                                            "tool_call",
                                            "Verifying Outcome",
                                            f"Executing Tool: {t_name}",
                                            f"Cross-checking live Web Portal DOM badge and SQLite database for {t_args.get('employee_id')}.",
                                        )
                                    else:
                                        _update_criteria(session_id, "determining_next")
                                        _append_step(
                                            session_id,
                                            "tool_call",
                                            "Select & Execute Tool",
                                            f"Executing Tool: {t_name}",
                                            f"Arguments: {t_args}",
                                        )

                        elif isinstance(msg, ToolMessage):
                            content_str = str(msg.content)
                            _update_criteria(session_id, "observing_result", "state_context")

                            if msg.name == "record_execution_plan":
                                continue

                            if "SESSION_LOCK_ERROR" in content_str:
                                _append_step(
                                    session_id,
                                    "failure_observed",
                                    "Observe Failure",
                                    f"Obstacle Detected ({msg.name})",
                                    content_str,
                                )
                            elif "RECOVERY_ACTION_SUCCESS" in content_str:
                                _update_criteria(session_id, "recovering_failures")
                                _append_step(
                                    session_id,
                                    "recovery_success",
                                    "Failure Resolved",
                                    f"Recovery Confirmed ({msg.name})",
                                    content_str,
                                )
                            elif "VERIFIED_SUCCESS" in content_str:
                                _update_criteria(session_id, "verifying_outcome")
                                _append_step(
                                    session_id,
                                    "verified",
                                    "Outcome Verified",
                                    f"Verification Evidence ({msg.name})",
                                    content_str,
                                )
                            else:
                                _append_step(
                                    session_id,
                                    "tool_observation",
                                    "Observe Result",
                                    f"Observation from {msg.name}",
                                    content_str,
                                )

            snapshot = AGENT_GRAPH.get_state(config)
            if not snapshot.next:
                with SESSIONS_LOCK:
                    AGENT_SESSIONS[session_id]["status"] = "completed"
                    AGENT_SESSIONS[session_id]["pending_hitl"] = None
                _update_criteria(session_id, "evidence_summary")
                break

            last_message = snapshot.values["messages"][-1]
            pending_tool_calls = getattr(last_message, "tool_calls", [])
            if not pending_tool_calls:
                current_input = None
                continue

            hitl_calls = [tc for tc in pending_tool_calls if tc["name"] in HITL_TOOL_NAMES]
            if not hitl_calls:
                current_input = None
                continue

            tc = hitl_calls[0]
            tool_name = tc["name"]
            tool_args = tc["args"]
            _update_criteria(session_id, "human_approval")

            if tool_name == "ask_human_clarification":
                cand_name = tool_args.get("candidate_name", "")
                cand_id = tool_args.get("candidate_id", "")
                prompt_title = "Ambiguous Identity Confirmation"
                prompt_question = tool_args.get(
                    "question",
                    f"Confirm proceeding with candidate {cand_name} ({cand_id})?",
                )
            else:
                emp_id = tool_args.get("employee_id", "UNKNOWN")
                emp_record = get_employee_by_id(emp_id)
                emp_display = f"{emp_record['name']} ({emp_id})" if emp_record else emp_id
                cand_name = emp_record["name"] if emp_record else ""
                cand_id = emp_id
                prompt_title = "Sensitive Action Approval (Data Vault Purge)"
                prompt_question = (
                    f"The workflow is requesting permission to permanently purge the personal data vault for {emp_display}. "
                    f"Do you approve this irreversible operation?"
                )

            with SESSIONS_LOCK:
                AGENT_SESSIONS[session_id]["status"] = "waiting_hitl"
                AGENT_SESSIONS[session_id]["pending_hitl"] = {
                    "tool_call_id": tc["id"],
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "title": prompt_title,
                    "question": prompt_question,
                    "candidate_name": cand_name,
                    "candidate_id": cand_id,
                }
            _append_step(
                session_id,
                "hitl_pause",
                "Human-in-the-Loop",
                f"Paused for Human Input: {prompt_title}",
                prompt_question,
            )
            return

    except Exception as exc:
        with SESSIONS_LOCK:
            if session_id in AGENT_SESSIONS:
                AGENT_SESSIONS[session_id]["status"] = "error"
                AGENT_SESSIONS[session_id]["error"] = str(exc)
        _append_step(session_id, "error", "Runtime Error", "Execution Error", str(exc))


@app.on_event("startup")
def on_startup() -> None:
    init_db(reset=False)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, q: Optional[str] = Query(default=None)):
    employees = get_all_employees(search_query=q)
    audit_logs = get_recent_audit_logs(limit=10)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "employees": employees,
            "search_query": q or "",
            "audit_logs": audit_logs,
        },
    )


@app.get("/employee/{employee_id}", response_class=HTMLResponse)
def employee_profile(request: Request, employee_id: str, message: Optional[str] = Query(default=None)):
    employee = get_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
    return templates.TemplateResponse(
        request=request,
        name="employee.html",
        context={
            "employee": employee,
            "flash_message": message,
        },
    )


@app.post("/employee/{employee_id}/deactivate")
def deactivate_employee(employee_id: str):
    employee = get_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")

    update_employee_status(employee_id, "inactive")
    return RedirectResponse(
        url=f"/employee/{employee_id}?message=Account+successfully+deactivated",
        status_code=303,
    )


@app.post("/employee/{employee_id}/activate")
def activate_employee(employee_id: str):
    employee = get_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")

    update_employee_status(employee_id, "active")
    update_vault_status(employee_id, "intact")
    reset_sso_session_locks()
    return RedirectResponse(
        url=f"/employee/{employee_id}?message=Account+re-activated+for+testing",
        status_code=303,
    )


@app.get("/api/employee/{employee_id}")
def api_get_employee(employee_id: str):
    employee = get_employee_by_id(employee_id)
    if not employee:
        raise HTTPException(status_code=404, detail=f"Employee '{employee_id}' not found")
    return employee


@app.get("/api/state")
def api_portal_state():
    return {
        "employees": get_all_employees(),
        "audit_logs": get_recent_audit_logs(limit=10),
    }


@app.post("/api/agent/start")
def api_start_agent(req: StartAgentRequest):
    if req.reset_db:
        init_db(reset=True)
    reset_sso_session_locks()

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    with SESSIONS_LOCK:
        AGENT_SESSIONS[session_id] = {
            "session_id": session_id,
            "task": req.task,
            "status": "running",
            "criteria": {
                "understanding_outcome": True,
                "determining_work": False,
                "planning_actions": False,
                "selecting_tools": False,
                "executing_actions": False,
                "observing_result": False,
                "determining_next": False,
                "recovering_failures": False,
                "state_context": True,
                "human_approval": False,
                "verifying_outcome": False,
                "evidence_summary": False,
            },
            "steps": [
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "type": "goal",
                    "phase": "User Goal",
                    "title": "Objective Received",
                    "detail": req.task,
                }
            ],
            "pending_hitl": None,
            "error": None,
        }

    initial_input = {"messages": [HumanMessage(content=req.task)]}
    worker = threading.Thread(
        target=_execute_agent_worker,
        args=(session_id, initial_input),
        daemon=True,
    )
    worker.start()
    return {"session_id": session_id, "status": "running"}


@app.get("/api/agent/status/{session_id}")
def api_agent_status(session_id: str):
    with SESSIONS_LOCK:
        session = AGENT_SESSIONS.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session_copy = dict(session)
        session_copy["steps"] = list(session["steps"])
        session_copy["criteria"] = dict(session["criteria"])

    return {
        "session": session_copy,
        "employees": get_all_employees(),
        "audit_logs": get_recent_audit_logs(limit=10),
    }


@app.post("/api/agent/hitl")
def api_agent_hitl(req: HitlDecisionRequest):
    with SESSIONS_LOCK:
        session = AGENT_SESSIONS.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        pending = session.get("pending_hitl")
        if not pending:
            raise HTTPException(status_code=400, detail="No pending HITL request for this session")

        session["status"] = "running"
        session["pending_hitl"] = None

    config = {"configurable": {"thread_id": req.session_id}}
    tool_name = pending["tool_name"]
    tool_call_id = pending["tool_call_id"]
    cand_name = pending.get("candidate_name", "")
    cand_id = pending.get("candidate_id", "")
    decision_raw = req.decision.strip()

    should_execute_tool = False
    injected_msg = None

    if tool_name == "ask_human_clarification":
        if decision_raw.lower() in ("approve", "y", "yes"):
            reply_text = (
                f"HUMAN_CONFIRMED: Yes, confirmed via Web Console. Proceed with offboarding "
                f"candidate '{cand_name}' (ID: '{cand_id}')."
            )
            _append_step(
                req.session_id,
                "hitl_decision",
                "Adapt to Human Input",
                "Human Confirmed Candidate Identity",
                reply_text,
            )
        elif decision_raw.lower() in ("deny", "n", "no"):
            reply_text = (
                f"HUMAN_DENIED: Do NOT offboard '{cand_name}' ({cand_id}). Abort the offboarding process and report back."
            )
            _append_step(
                req.session_id,
                "hitl_decision",
                "Adapt to Human Input",
                "Human Rejected Candidate Identity",
                reply_text,
            )
        else:
            reply_text = f"HUMAN_CLARIFICATION: {decision_raw}"
            _append_step(
                req.session_id,
                "hitl_decision",
                "Adapt to Human Input",
                "Human Provided Custom Instruction",
                reply_text,
            )

        injected_msg = ToolMessage(content=reply_text, name=tool_name, tool_call_id=tool_call_id)

    elif tool_name == "delete_user_data_vault":
        if decision_raw.lower() in ("approve", "y", "yes"):
            should_execute_tool = True
            _append_step(
                req.session_id,
                "hitl_decision",
                "Human Approved",
                "Vault Purge Authorized",
                f"Approved permanent data vault purge for {cand_name} ({cand_id}).",
            )
        else:
            reply_text = (
                f"USER_DENIED_PERMISSION: The human operator denied permission to delete the sensitive data vault "
                f"for {cand_name} ({cand_id}). Do not retry deleting the vault; proceed directly to verify_deactivation "
                f"and note in the final summary that the data vault was retained per human instruction."
            )
            _append_step(
                req.session_id,
                "hitl_decision",
                "Adapt to Human Input",
                "Vault Purge Denied — Adapting Plan",
                "Instructed agent to retain data vault and proceed to verification.",
            )
            injected_msg = ToolMessage(content=reply_text, name=tool_name, tool_call_id=tool_call_id)

    if not should_execute_tool and injected_msg is not None:
        AGENT_GRAPH.update_state(
            config,
            {"messages": [injected_msg]},
            as_node="tools_node",
        )

    worker = threading.Thread(
        target=_execute_agent_worker,
        args=(req.session_id, None),
        daemon=True,
    )
    worker.start()
    return {"status": "resumed"}


@app.post("/reset-db")
def reset_database():
    init_db(reset=True)
    reset_sso_session_locks()
    return RedirectResponse(url="/", status_code=303)
