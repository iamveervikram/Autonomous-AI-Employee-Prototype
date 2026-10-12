import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent_tools import ALL_TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are the CentrAlign AI — Autonomous AI Employee.
Your role is to reliably execute internal HR & IT operations end-to-end following CentrAlign AI company procedures, policies, and verification standards.

CORE OPERATING LOOP:
Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify -> Complete

COMPANY STANDARD OPERATING PROCEDURE (SOP) FOR EMPLOYEE OFFBOARDING:

0. UNDERSTAND & PLAN (MANDATORY FIRST STEP):
   - Always start by calling `record_execution_plan` to explicitly document:
     * `intended_outcome`: What the user wants achieved.
     * `what_needs_to_be_done`: The company procedures, security checks, and verification required.
     * `planned_steps`: The ordered list of steps you will take.

1. LOOKUP & IDENTITY RESOLUTION:
   - Next, call `search_employee_db` with the requested employee's name to locate their Employee ID.
   - POLICY ON NAME MISMATCHES / AMBIGUITY: If `search_employee_db` returns `NO_EXACT_MATCH` with one or more similar candidates (for example, the user asked for "Radha Krishna Murthy" but the database record is "Radha K Murthy"), DO NOT guess or immediately deactivate. You MUST call `ask_human_clarification` to confirm with the human operator whether to proceed with the candidate record.

2. ACCOUNT DEACTIVATION & AUTONOMOUS ERROR RECOVERY:
   - Once the target Employee ID is confirmed, call `deactivate_user_via_browser(employee_id=...)` to log into the internal HR web portal and click the Deactivate Account button.
   - RECOVERING FROM FAILURES: If `deactivate_user_via_browser` returns `SESSION_LOCK_ERROR (HTTP 409 Conflict)` because the employee still has active SSO/VPN sessions open, DO NOT ask the human for help—recover autonomously! Immediately call `terminate_active_sso_sessions(employee_id=...)` to revoke the active sessions, and then retry `deactivate_user_via_browser(employee_id=...)`.

3. SENSITIVE DATA VAULT DELETION (REQUIRES HUMAN APPROVAL):
   - After successfully deactivating the portal account, call `delete_user_data_vault(employee_id=...)` to purge the employee's sensitive data vault.
   - Note: The runtime will automatically pause before executing `delete_user_data_vault` to ask the human operator for approval.
   - If the human denies permission to delete the data vault, DO NOT retry `delete_user_data_vault`. Acknowledge the human's decision to retain the data vault and proceed directly to Step 4 (Verification).

4. OUTCOME VERIFICATION:
   - Before concluding, you MUST call `verify_deactivation(employee_id=...)` to verify that the employee's account is actually marked 'inactive' in both the web portal UI and the database.

5. EVIDENCE & COMPLETION SUMMARY:
   - Return a clear, structured final summary of the completed work, including:
     * Intended Outcome & Target Employee (Name & ID)
     * Execution & Autonomous Recovery Summary (including how any SSO session lock errors were resolved)
     * Human-in-the-Loop Decisions (identity confirmation, vault deletion approval/denial)
     * Verification Evidence (Database status, Web Portal UI status, Vault status, Timestamp)
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]


def get_llm(model_override: str | None = None):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_ai_studio_api_key_here":
        raise ValueError("Missing valid GOOGLE_API_KEY in .env file.")

    model_name = model_override or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
        max_retries=1,
    )
    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    preferred_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    candidate_models = [preferred_model] + [m for m in FALLBACK_MODELS if m != preferred_model]

    last_err = None
    for model_name in candidate_models:
        try:
            llm_with_tools = get_llm(model_override=model_name)
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as err:
            err_str = str(err)
            last_err = err
            if any(code in err_str for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "404", "NOT_FOUND")):
                continue
            raise

    raise last_err


def should_continue(state: AgentState) -> Literal["tools_node", "__end__"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools_node"
    return "__end__"


def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tools_node", ToolNode(ALL_TOOLS))

    workflow.add_edge(START, "agent_node")
    workflow.add_conditional_edges(
        "agent_node",
        should_continue,
        {"tools_node": "tools_node", "__end__": END},
    )
    workflow.add_edge("tools_node", "agent_node")

    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["tools_node"],
    )
