# CentrAlign AI — Autonomous AI Employee

This repository implements the **CentrAlign AI — Autonomous AI Employee** prototype for the CentrAlign AI Engineering Assignment. It demonstrates the complete autonomous execution loop:

> **Goal → Understand → Plan → Execute → Observe → Adapt / Recover → Verify → Complete**

---

## 1. Architecture Overview

The system is architected as two decoupled layers that mirror how an autonomous AI operator interacts with real enterprise infrastructure:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                   1. Autonomous AI Employee Runtime                          │
│                                                                              │
│  User Goal ──► [LangGraph State Machine + MemorySaver Checkpointer]          │
│                       │                                   ▲                  │
│                       ▼                                   │                  │
│             ┌───────────────────┐               ┌───────────────────┐        │
│             │    agent_node     │  Tool Calls   │    tools_node     │        │
│             │ (Gemini + Fallback│ ────────────► │  (Pydantic Tools) │        │
│             │   SOP Reasoning)  │ ◄──────────── │                   │        │
│             └───────────────────┘  Observations └───────────────────┘        │
│                                                           │                  │
│                                      Deterministic HITL Interrupt Gate       │
│                                  (Ambiguous Identity & Data Vault Purge)     │
└───────────────────────────────────────────────────────────┼──────────────────┘
                                                            │
                        ┌───────────────────────────────────┴──────────────┐
                        ▼                                                  ▼
┌───────────────────────────────────────────┐    ┌─────────────────────────────┐
│  2A. Browser Computer-Use (Playwright)    │    │ 2B. Direct Backend / DB     │
│  - Navigates FastAPI HR Portal (/employee)│    │ - SQLite (hr_system.db)     │
│  - Clicks '#deactivate-btn'               │    │ - Exact + Fuzzy Name Search │
│  - Reads live DOM status badges           │    │ - SSO Session Remediation   │
│                                           │    │ - Security Audit Logging    │
└───────────────────────────────────────────┘    └─────────────────────────────┘
```

### Core Components
1. **Stateful Agent Runtime ([`agent_graph.py`](agent_graph.py))**:
   - Built on **LangGraph** (`StateGraph`) with `MemorySaver` thread checkpointing.
   - Orchestrates a cyclical loop between `agent_node` (Google Gemini LLM reasoning) and `tools_node` (tool execution), pausing deterministically via `interrupt_before=["tools_node"]` whenever human input or authorization is required.
2. **Tool & Connector Layer ([`agent_tools.py`](agent_tools.py))**:
   - Exposes 7 structured, Pydantic-validated tools (`record_execution_plan`, `search_employee_db`, `ask_human_clarification`, `deactivate_user_via_browser`, `terminate_active_sso_sessions`, `delete_user_data_vault`, `verify_deactivation`).
3. **Enterprise HR Portal & Live Operator Console ([`mock_system/main.py`](mock_system/main.py), [`mock_system/database.py`](mock_system/database.py))**:
   - A **FastAPI + SQLite** application that serves both the target company HR system (`/employee/{id}`) and a real-time web console (`/`) displaying the 12-point capability tracker, execution trace, Human-in-the-Loop approval banners, and immutable security audit logs.

---

## 2. Setup & Running the Prototype

### 1. Install Dependencies
```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure `.env`
```powershell
Copy-Item .env.example .env
```
Add your Google AI Studio key in `.env`:
```env
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-3.6-flash
PORTAL_BASE_URL=http://localhost:8000
HEADLESS_BROWSER=false
```

### 3. Start the Unified Web Application
```powershell
uvicorn mock_system.main:app --reload --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser to run scenarios, watch the live 12-point capability checklist, approve/deny Human-in-the-Loop prompts, and inspect the Employee Directory and Audit Logs in real time.

*(Optional CLI Mode)*: You can also run the workflow from the terminal via:
```powershell
python main_agent.py --reset-db
```
