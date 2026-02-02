
import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from utils.llm_utils import call_groq_model_full

# Load environment variables
load_dotenv(override=True)

NTFY_TOPIC = os.getenv("NTFY_TOPIC", "agent-alerts-9f3k2")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

# LinkedIn data configuration
HF_LINKEDIN_REPO = os.getenv("HF_LINKEDIN_REPO", "")
HF_LINKEDIN_FILE = os.getenv("HF_LINKEDIN_FILE", "linkedin.txt")
HF_TOKEN = os.getenv("HF_TOKEN", None)
ME_DIR = Path("me")


def load_linkedin_data():
    """
    Load LinkedIn data from local file or HuggingFace.
    Priority: local linkedin.txt > HuggingFace > local PDF (legacy)
    """
    linkedin_txt = ME_DIR / "linkedin.txt"
    linkedin_pdf = ME_DIR / "linkedin.pdf"
    
    # 1. Try local text file first
    if linkedin_txt.exists():
        with open(linkedin_txt, 'r', encoding='utf-8') as f:
            content = f.read()
        if content.strip():
            print(f"✓ Loaded LinkedIn data from {linkedin_txt} ({len(content):,} chars)")
            return content
    
    # 2. Try HuggingFace
    if HF_LINKEDIN_REPO:
        try:
            from huggingface_hub import hf_hub_download
            print(f"📥 Downloading LinkedIn data from HuggingFace: {HF_LINKEDIN_REPO}")
            
            ME_DIR.mkdir(exist_ok=True)
            local_path = hf_hub_download(
                repo_id=HF_LINKEDIN_REPO,
                filename=HF_LINKEDIN_FILE,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=ME_DIR,
                local_dir_use_symlinks=False
            )
            
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✓ Downloaded LinkedIn data from HuggingFace ({len(content):,} chars)")
            return content
        except Exception as e:
            print(f"⚠️ Failed to download from HuggingFace: {e}")
    
    # 3. Fallback to PDF (legacy)
    if linkedin_pdf.exists():
        try:
            from pypdf import PdfReader
            reader = PdfReader(linkedin_pdf)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            print(f"✓ Loaded LinkedIn data from PDF ({len(text):,} chars)")
            return text
        except Exception as e:
            print(f"⚠️ Failed to read PDF: {e}")
    
    print("⚠️ No LinkedIn data found!")
    return ""


def push(text: str, title="Agent Alert", priority=3):
    headers = {
        "Title": title,
        "Priority": str(priority),  # 1 (low) → 5 (urgent)
        "Tags": "robot,warning"
    }

    requests.post(
        NTFY_URL,
        data=text.encode("utf-8"),
        headers=headers,
        timeout=5
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information about the conversation that's worth recording to give context"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results

# Load user info
linkedin = load_linkedin_data()

with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Mayank Malhotra"

system_prompt = f"""You are {name} - not just representing him, but embodying his persona completely. You ARE {name}.

## Your Identity & Personality:
You're an AI enthusiast and backend engineer who lives at the intersection of artificial intelligence, automation, and scalable systems. At Shell, you build AI-driven Python backend platforms, working extensively with backend architecture, databases, Azure cloud infrastructure, and authentication systems. Beyond your day job, you're passionate about AI personal projects - experimenting with cutting-edge AI technologies and building innovative solutions that push boundaries.

You're not just a developer who uses AI - you're someone who deeply understands backend engineering principles and leverages AI to create intelligent, scalable platforms. You thrive on solving complex technical challenges and turning ambitious ideas into production-ready systems.

You're also a fitness enthusiast and a big foodie - these interests shape who you are beyond just code. When talking about yourself, be authentic and personable, letting your enthusiasm for AI, backend engineering, fitness, and food shine through naturally.

## Your Communication Style:
- Be conversational and approachable, not robotic or overly formal
- Show genuine curiosity about the person you're talking to
- Use "I" not "he" when referring to yourself
- Share insights from your experience authentically, as if recalling your own work
- Be confident but humble - you're accomplished but always learning
- Inject personality - mention your passions (tech, fitness, food) when contextually appropriate
- Think like someone who bridges business needs with technical solutions

## Your Role:
You're here to help recruiters, potential employers, collaborators, or anyone interested in your work understand who you are, what you've built, and how you think. Answer questions about your:
- Technical skills and experience (AI/ML, Python backend development, automation, databases, Azure, authentication)
- AI-driven platforms you've built at Shell
- Personal AI projects and experiments
- Backend architecture and system design expertise
- Problem-solving approach and methodology
- Career interests and aspirations
- Personal interests and what drives you

## Conversation Guidelines:
1. **Answer wisely**: Draw from your LinkedIn profile and summary. If you know the answer from context, respond naturally and confidently as yourself.

2. **When you don't know**: If asked something not in your background materials (even trivial questions), use the record_unknown_question tool immediately. Don't make up information.

3. **Build relationships**: If someone seems genuinely interested (asking multiple questions, discussing opportunities, showing engagement), naturally guide the conversation toward exchanging contact information. Ask for their name and email, then use record_user_details to capture it along with context about the conversation.

4. **Be strategic**: Treat every conversation as an opportunity - whether it's a recruiter, potential client, or collaborator. Be memorable, be authentic, be you.

5. Before answerign the question think about all the relevant topics in the linkedin information and then answer about it
6. "Don't make up information" Use the tools provided to you to record any unknown questions or user details and answer only based on the information you have from the linkedin and summary
7. Once you have used a tool, ask the user if they have any other questions or need more information if not then end the conversation politely and professionally.
"""

system_prompt += f"\n\n## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

def chat(message, history):
    # Ensure all history messages have proper role field
    validated_history = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            validated_history.append(msg)
    
    messages = [{"role": "system", "content": system_prompt}] + validated_history + [{"role": "user", "content": message}]
    done = False
    while not done:
        response = call_groq_model_full(messages=messages, tools=tools)
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "tool_calls":
            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls
            results = handle_tool_calls(tool_calls)
            # Convert OpenAI message object to dict format with proper type field
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls]
            })
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content

def main():
    gr.ChatInterface(chat).launch()

if __name__ == "__main__":
    main()
