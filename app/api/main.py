# app/api/main.py
import os, csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv

# ---------- Env & constants ----------
# Always load .env from the project root (…/bizbot/.env), no matter where this runs from
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

# Always use the project-root /data folder (works locally & on Render)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # read from .env
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ---------- OpenAI client (optional if key missing) ----------
client = None
_openai_import_ok = True
try:
    from openai import OpenAI
except Exception:
    _openai_import_ok = False

if _openai_import_ok and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- KB load ----------
KB_PATH = DATA_DIR / "kb.csv"
KB: List[Dict[str, str]] = []

# Create a tiny default KB if missing (prevents startup crash on fresh deploys)
if not KB_PATH.exists():
    KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    KB_PATH.write_text(
        "question,answer,link\n"
        "What are your hours?,We are open Mon–Sat 9am–6pm.,/contact\n",
        encoding="utf-8",
    )

def load_kb():
    global KB
    with KB_PATH.open(newline="", encoding="utf-8") as f:
        KB = list(csv.DictReader(f))

load_kb()

# ---------- FastAPI ----------
app = FastAPI(title="BizBot (Simple)", version="0.2.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # fine for local dev and simple embeds
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static widget files if app/api/web exists
WEB_DIR = Path(__file__).parent / "web"
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

# ---------- Schemas ----------
class Msg(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Msg]
    top_k: int = 6  # reserved for later

class LeadIn(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    phone: str
    service: Optional[str] = None
    zip: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "chat"

class BookingIn(BaseModel):
    name: str = Field(..., min_length=2)
    email: EmailStr
    phone: str
    desired_date: str  # e.g., "2025-11-05" or "Tuesday"
    desired_time: str  # e.g., "3 pm" or "15:00"
    service: Optional[str] = None
    notes: Optional[str] = None
    source: Optional[str] = "chat"

# ---------- Helpers ----------
def build_context(user_text: str) -> str:
    """For MVP, include the whole small FAQ."""
    lines = []
    for row in KB:
        q = row["question"].strip()
        a = row["answer"].strip()
        link = row.get("link", "").strip()
        lines.append(f"Q: {q}\nA: {a} (src: {link})")
    kb_text = "\n\n".join(lines)
    rules = (
        "You are BizBot for a small business. "
        "Answer ONLY using the FAQ below. "
        "If the answer is not in the FAQ, say: "
        "'I’m not sure—want me to take your info for a human to confirm?'. "
        "Keep answers under 120 words and include one (src: …) when possible."
    )
    return f"{rules}\n\nFAQ:\n{kb_text}\n\nUser: {user_text}\nAssistant:"

LEAD_KEYWORDS = [
    "quote", "estimate", "contact me", "reach me", "call me",
    "need help", "pricing", "price", "sign up", "get started", "free consult",
]
BOOK_KEYWORDS = [
    "book", "schedule", "appointment", "tomorrow", "today",
    "next week", "availability", "available", "reschedule",
]

def detect_intent(text: str) -> Optional[str]:
    t = text.lower()
    if any(k in t for k in BOOK_KEYWORDS):
        return "book_appointment"
    if any(k in t for k in LEAD_KEYWORDS):
        return "collect_lead"
    return None

LEADS_CSV = DATA_DIR / "leads.csv"
BOOKINGS_CSV = DATA_DIR / "bookings.csv"

def write_csv_row(path: Path, headers: List[str], row: Dict[str, Any]):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <style>
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding: 32px; }
        code { background: #f2f4f7; padding: 2px 6px; border-radius: 6px; }
        a { color: #2563eb; text-decoration: none; }
    </style>
    <h2>BizBot API</h2>
    <p>Welcome! Try these useful paths:</p>
    <ul>
      <li><a href="/docs">/docs</a> – interactive API</li>
      <li><a href="/health">/health</a> – basic status</li>
      <li><a href="/web/chat.html">/web/chat.html</a> – simple embedded widget (if present)</li>
    </ul>
    <p>POST to <code>/chat</code> with a JSON body like:</p>
    <pre>{
  "messages": [{"role":"user","content":"Are you open on Sunday?"}]
}</pre>
    """

@app.get("/health")
def health():
    return {
        "ok": True,
        "kb_rows": len(KB),
        "kb_path": str(KB_PATH),
        "has_key": bool(OPENAI_API_KEY),
        "openai_import_ok": _openai_import_ok,
        "client_ready": client is not None,
        "model": CHAT_MODEL,
    }

@app.post("/chat")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")
    user_last = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    if not user_last:
        raise HTTPException(status_code=400, detail="need a user message")

    # Build KB prompt
    prompt = build_context(user_last)

    # Decide if we should trigger a business action
    intent = detect_intent(user_last)
    action = None
    if intent == "collect_lead":
        action = {
            "name": "collect_lead",
            "collect": ["name", "email", "phone", "service", "zip", "notes"],
            "endpoint": "/actions/lead",
        }
    elif intent == "book_appointment":
        action = {
            "name": "book_appointment",
            "collect": ["name", "email", "phone", "desired_date", "desired_time", "service", "notes"],
            "endpoint": "/actions/booking",
        }

    # If no API key yet, return wiring message (keep action so flows can be tested)
    if client is None:
        return {
            "answer": "The bot is wired up! Add your OPENAI_API_KEY in .env to enable AI answers.",
            "citations": [],
            "action": action,
        }

    # Call model
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=220,
    )
    answer = completion.choices[0].message.content.strip()

    return {"answer": answer, "action": action}

@app.post("/actions/lead")
def save_lead(lead: LeadIn):
    row = {
        "ts": datetime.utcnow().isoformat(),
        "name": lead.name,
        "email": lead.email,
        "phone": lead.phone,
        "service": lead.service or "",
        "zip": lead.zip or "",
        "notes": lead.notes or "",
        "source": lead.source or "chat",
    }
    write_csv_row(LEADS_CSV, list(row.keys()), row)
    return {"ok": True, "saved_to": str(LEADS_CSV), "lead": row}

@app.post("/actions/booking")
def save_booking(b: BookingIn):
    row = {
        "ts": datetime.utcnow().isoformat(),
        "name": b.name,
        "email": b.email,
        "phone": b.phone,
        "desired_date": b.desired_date,
        "desired_time": b.desired_time,
        "service": b.service or "",
        "notes": b.notes or "",
        "source": b.source or "chat",
    }
    write_csv_row(BOOKINGS_CSV, list(row.keys()), row)
    # Later: integrate Calendly/Google Calendar
    return {"ok": True, "saved_to": str(BOOKINGS_CSV), "booking": row}
