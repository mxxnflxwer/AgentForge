import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.reranker import LocalSemanticReranker

ranker = LocalSemanticReranker()
query = "What are the symptoms?"

plan_chunk = {
    "section_title": "Plan",
    "content": "Plan\nRecommend coronary CT angiography for further risk stratification. Start atorvastatin 40 mg daily and low-dose aspirin. Follow up in 4 weeks.",
    "similarity_score": 0.5327,
}

cc_chunk = {
    "section_title": "Chief Complaint",
    "content": "Chief Complaint\nPatient presents with intermittent chest discomfort and shortness of breath on exertion for the past 3 weeks.",
    "similarity_score": 0.5158,
}

print("Plan score:", ranker.compute_chunk_score(query, plan_chunk))
tokens = ["symptoms"]
print("Plan lexical:", ranker.compute_lexical_score(tokens, plan_chunk["content"].lower(), plan_chunk["section_title"].lower()))
print("Plan intent:", ranker.compute_intent_score(tokens, plan_chunk["content"].lower(), plan_chunk["section_title"].lower()))

print("\nCC score:", ranker.compute_chunk_score(query, cc_chunk))
print("CC lexical:", ranker.compute_lexical_score(tokens, cc_chunk["content"].lower(), cc_chunk["section_title"].lower()))
print("CC intent:", ranker.compute_intent_score(tokens, cc_chunk["content"].lower(), cc_chunk["section_title"].lower()))
