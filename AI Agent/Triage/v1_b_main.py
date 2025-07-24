import json
import os
import datetime
from dataclasses import dataclass
from typing import Dict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

mock_knowledge_base = {
    "password reset": "To reset your password, click 'Forgot Password' on the login page.",
    "billing issue": "Please ensure your billing details are updated. Contact billing@generic.com."
}

@dataclass
class Ticket:
    user_id: str
    timestamp: datetime.datetime
    channel: str
    description: str

@dataclass
class ProcessedTicket:
    raw_ticket: Ticket
    metadata: Dict
    intent: Optional[str] = None
    entities: Optional[Dict] = None
    sentiment: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    assigned_to: Optional[str] = None
    auto_response: Optional[str] = None
    resolution: Optional[str] = None

class IntakeAgent:
    def process(self, ticket: Ticket) -> Dict:
        print("[IntakeAgent] Processing metadata...")
        return {
            "timestamp": ticket.timestamp.isoformat(),
            "channel": ticket.channel,
            "length": len(ticket.description)
        }

class UnderstandingAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def analyze(self, ticket: Ticket) -> Dict:
        print("[UnderstandingAgent] Using LLM to analyze intent, entities, sentiment, and category...")
        prompt = f"""
        Analyze the following support ticket and return ONLY a valid JSON object with these keys:
        - intent (e.g., "password reset", "billing issue")
        - entities (a dictionary with a "keywords" list)
        - sentiment (e.g., "positive", "neutral", "frustrated")
        - category (e.g., "Technical", "Billing")

        Support ticket:
        {ticket.description}
        """
        result = self.llm.invoke([HumanMessage(content=prompt.strip())])

        try:
            content = result.content.strip()
            if content.startswith("```json"):
                content = content[7:].strip()
            if content.startswith("```"):
                content = content[3:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print("[Error parsing LLM response]", e)
            print("Raw content was:\n", result.content)
            return {}

class PrioritizationAgent:
    def prioritize(self, entities: Dict, sentiment: str) -> str:
        print("[PrioritizationAgent] Determining priority level...")
        keywords = entities.get("keywords", [])
        if "critical" in keywords or sentiment.lower() == "frustrated":
            return "High"
        return "Normal"

class RoutingAgent:
    def route(self, category: str, priority: str) -> str:
        print("[RoutingAgent] Routing based on category and priority...")
        if category == "Billing":
            return "Billing Team"
        if priority == "High":
            return "Tier 2 Support"
        return "Tier 1 Support"

class KnowledgeAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def retrieve_response(self, intent: str) -> Optional[str]:
        print("[KnowledgeAgent] Fetching knowledge base response...")
        return mock_knowledge_base.get(intent)

class ResolutionAgent:
    def resolve(self, entities: Dict) -> Optional[str]:
        print("[ResolutionAgent] Attempting automated resolution...")
        if entities.get("intent") == "password reset":
            return "Triggered password reset API."
        return None

class FeedbackAgent:
    def monitor(self, processed_ticket: ProcessedTicket) -> None:
        print("[FeedbackAgent] Monitoring final outcome...")

class TriageSystem:
    def __init__(self, llm: ChatOpenAI):
        self.intake = IntakeAgent()
        self.understanding = UnderstandingAgent(llm)
        self.prioritization = PrioritizationAgent()
        self.routing = RoutingAgent()
        self.knowledge = KnowledgeAgent(llm)
        self.resolution = ResolutionAgent()
        self.feedback = FeedbackAgent()

    def handle_ticket(self, ticket: Ticket) -> ProcessedTicket:
        metadata = self.intake.process(ticket)
        understanding = self.understanding.analyze(ticket)
        priority = self.prioritization.prioritize(understanding.get("entities", {}), understanding.get("sentiment", ""))
        assigned_to = self.routing.route(understanding.get("category", ""), priority)
        auto_response = self.knowledge.retrieve_response(understanding.get("intent", ""))
        resolution = self.resolution.resolve(understanding.get("entities", {}))

        processed = ProcessedTicket(
            raw_ticket=ticket,
            metadata=metadata,
            intent=understanding.get("intent"),
            entities=understanding.get("entities"),
            sentiment=understanding.get("sentiment"),
            category=understanding.get("category"),
            priority=priority,
            assigned_to=assigned_to,
            auto_response=auto_response,
            resolution=resolution
        )

        self.feedback.monitor(processed)
        return processed

if __name__ == '__main__':
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    triage = TriageSystem(llm)

    new_ticket = Ticket(
        user_id="cust123",
        timestamp=datetime.datetime.now(),
        channel="email",
        description="I can't log in and need a password reset urgently!"
    )

    result = triage.handle_ticket(new_ticket)
    print("\nProcessed Ticket Result:\n")
    print(result)

    print("\Intake Agent:\n")
    intake_agent = IntakeAgent()
    metadata = intake_agent.process(new_ticket)

    # Print each field individually
    print("Timestamp:", metadata["timestamp"])
    print("Channel:", metadata["channel"])
    print("Description Length:", metadata["length"])

