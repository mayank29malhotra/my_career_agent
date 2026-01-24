
import os
import json
import requests
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr
from utils.llm_utils import call_groq_model_full

# Load environment variables
load_dotenv(override=True)

NTFY_TOPIC = os.getenv("NTFY_TOPIC", "agent-alerts-9f3k2")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"


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
reader = PdfReader("me/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

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
            # Convert OpenAI message object to dict format
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [{"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tool_calls]
            })
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content

def main():
    gr.ChatInterface(chat).launch(share=True)

if __name__ == "__main__":
    main()
