
import json
from agents.rag_agent import RAGAgent

agent = RAGAgent()
data = json.load(open("data/benchmarks/qa_eval.json"))

score = 0
for d in data:
    if d["answer"].lower() in agent.answer(d["question"]).lower():
        score += 1

print(f"Accuracy: {100*score/len(data):.2f}%")
