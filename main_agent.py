import argparse
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_graph import build_agent_graph
from agent_tools import HITL_TOOL_NAMES, reset_sso_session_locks
from mock_system.database import get_employee_by_id, init_db

load_dotenv()


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


def print_banner() -> None:
    print("=" * 72)
    print("  CentrAlign AI — Autonomous AI Employee")
    print("  Loop: Goal -> Understand -> Plan -> Execute -> Observe -> Adapt -> Verify")
    print("=" * 72)


def run_autonomous_task(user_request: str, reset_db: bool = False) -> None:
    init_db(reset=reset_db)
    reset_sso_session_locks()
    graph = build_agent_graph()

    thread_id = f"offboard-session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print_banner()
    print(f"\n[USER GOAL]: {user_request}")
    print(f"[SESSION ID]: {thread_id}\n")

    current_input: Any = {"messages": [HumanMessage(content=user_request)]}

    while True:
        for event in graph.stream(current_input, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if not isinstance(node_output, dict):
                    continue
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        text = _extract_text(msg.content).strip()
                        if text:
                            print(f"\n[AI EMPLOYEE]:\n{text}\n")
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                print(f"  -> [PLANNING TOOL CALL]: {tc['name']}({tc['args']})")
                    elif isinstance(msg, ToolMessage):
                        print(f"  <- [TOOL OBSERVATION ({msg.name})]:\n     {msg.content}\n")

        snapshot = graph.get_state(config)
        if not snapshot.next:
            print("\n" + "=" * 72)
            print("  WORKFLOW COMPLETED")
            print("=" * 72)
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

        injected_tool_messages = []
        should_execute_tools_node = True

        for tc in hitl_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_call_id = tc["id"]

            if tool_name == "ask_human_clarification":
                question = tool_args.get("question", "Please confirm how to proceed.")
                cand_name = tool_args.get("candidate_name", "")
                cand_id = tool_args.get("candidate_id", "")

                print("\n" + "!" * 72)
                print("  [HUMAN-IN-THE-LOOP: IDENTITY / AMBIGUITY CLARIFICATION REQUIRED]")
                print("!" * 72)
                print(f"  Agent Question: {question}")
                if cand_name or cand_id:
                    print(f"  Candidate Record in DB: {cand_name} ({cand_id})")

                user_reply = input(
                    "\n  >> Confirm proceeding with this candidate? (Y/N or type custom instruction): "
                ).strip()

                should_execute_tools_node = False
                if user_reply.lower() in ("y", "yes", ""):
                    reply_text = (
                        f"HUMAN_CONFIRMED: Yes, confirmed. Proceed with offboarding candidate "
                        f"'{cand_name}' (ID: '{cand_id}')."
                    )
                    print("  [HITL DECISION]: Approved candidate match. Resuming autonomous execution...\n")
                elif user_reply.lower() in ("n", "no"):
                    reply_text = (
                        f"HUMAN_DENIED: Do NOT offboard '{cand_name}' ({cand_id}). "
                        f"Abort the offboarding process and report back."
                    )
                    print("  [HITL DECISION]: Denied candidate match. Informing agent...\n")
                else:
                    reply_text = f"HUMAN_CLARIFICATION: {user_reply}"
                    print("  [HITL DECISION]: Sent custom clarification to agent...\n")

                injected_tool_messages.append(
                    ToolMessage(
                        content=reply_text,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )

            elif tool_name == "delete_user_data_vault":
                emp_id = tool_args.get("employee_id", "UNKNOWN")
                emp_record = get_employee_by_id(emp_id)
                emp_display = f"{emp_record['name']} ({emp_id})" if emp_record else emp_id

                print("\n" + "!" * 72)
                print("  [HUMAN-IN-THE-LOOP: SENSITIVE ACTION APPROVAL GATE]")
                print("!" * 72)
                print(f"  Action Requested : delete_user_data_vault (Permanent Data Purge)")
                print(f"  Target Employee  : {emp_display}")
                approval = input(
                    f"\n  >> Agent requested permission to delete data for {emp_display}. Approve? (Y/N): "
                ).strip()

                if approval.lower() in ("y", "yes"):
                    print("  [HITL DECISION]: Approved! Executing data vault purge...\n")
                    should_execute_tools_node = True
                else:
                    print("  [HITL DECISION]: Denied! Instructing agent to skip vault deletion and adapt...\n")
                    should_execute_tools_node = False
                    injected_tool_messages.append(
                        ToolMessage(
                            content=(
                                f"USER_DENIED_PERMISSION: The human operator denied permission to delete the "
                                f"sensitive data vault for {emp_display}. Do not retry deleting the vault; "
                                f"proceed with verifying the account deactivation and note in the final summary "
                                f"that the data vault was retained per human instruction."
                            ),
                            name=tool_name,
                            tool_call_id=tool_call_id,
                        )
                    )

        if should_execute_tools_node:
            current_input = None
        else:
            graph.update_state(
                config,
                {"messages": injected_tool_messages},
                as_node="tools_node",
            )
            current_input = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CentrAlign AI — Autonomous AI Employee"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task instruction for the AI employee",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Reset the SQLite HR database to initial seed state before running.",
    )
    args = parser.parse_args()

    if args.task:
        task_prompt = args.task
    else:
        print_banner()
        print("\nAvailable Demo Scenarios:")
        print("  1. Exact Match Offboarding : 'Please offboard John Doe.'")
        print("  2. Fuzzy Name Match + HITL : 'Please offboard Radha Krishna Murthy.' (DB has 'Radha K Murthy')")
        print("  3. Custom Request          : Type any instruction\n")
        choice = input("Select scenario (1/2) or type a custom request [default: 1]: ").strip()
        if choice == "2":
            task_prompt = "Please offboard Radha Krishna Murthy."
        elif choice in ("1", ""):
            task_prompt = "Please offboard John Doe."
        else:
            task_prompt = choice

    try:
        run_autonomous_task(task_prompt, reset_db=args.reset_db)
    except ValueError as err:
        print(f"\n[CONFIGURATION ERROR]: {err}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED]: Execution stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
