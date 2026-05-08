# fraud_agent_ui.py
# Run from SageMaker terminal: python fraud_agent_ui.py
# Or from a notebook cell:    !python fraud_agent_ui.py &
# Access via the Gradio URL printed at startup (SageMaker Studio proxies port 7860).

import os
import logging
import boto3
import sagemaker
import gradio as gr
from sagemaker import get_execution_role
from strands import Agent, tool as strands_tool
from strands.models import BedrockModel
from strands_tools import mem0_memory

logging.getLogger("mem0").setLevel(logging.CRITICAL)

# Auth (same pattern as notebook)
sess       = sagemaker.Session()
role       = get_execution_role()
AWS_REGION = sess.boto_region_name
os.environ["AWS_REGION"]         = AWS_REGION
os.environ["AWS_DEFAULT_REGION"] = AWS_REGION

MODEL_ID        = "us.anthropic.claude-3-haiku-20240307-v1:0"
EMBED_MODEL_ID  = "amazon.titan-embed-text-v2:0"
RERANK_MODEL_ID = "cohere.rerank-v3-5:0"
RERANK_MODEL_ARN = f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{RERANK_MODEL_ID}"

KB_ID = os.environ.get("KNOWLEDGE_BASE_ID") or os.environ.get("STRANDS_KNOWLEDGE_BASE_ID") or "FARSQGTONR"
os.environ["KNOWLEDGE_BASE_ID"]         = KB_ID
os.environ["STRANDS_KNOWLEDGE_BASE_ID"] = KB_ID
os.environ["MIN_SCORE"]                 = "0.2"
os.environ["RETRIEVE_ENABLE_METADATA_DEFAULT"] = "true"

os.environ["MEM0_LLM_PROVIDER"]   = "aws_bedrock"
os.environ["MEM0_LLM_MODEL"]      = MODEL_ID
os.environ["MEM0_EMBED_PROVIDER"] = "aws_bedrock"
os.environ["MEM0_EMBED_MODEL"]    = EMBED_MODEL_ID

llm                   = BedrockModel(model_id=MODEL_ID, region_name=AWS_REGION)
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


def _bedrock_rerank(query_text, text_sources, num_results=3):
    resp = bedrock_agent_runtime.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query_text}}],
        sources=[
            {"type": "INLINE",
             "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": s}}}
            for s in text_sources
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": num_results,
                "modelConfiguration": {"modelArn": RERANK_MODEL_ARN},
            },
        },
    )
    return [{"index": r["index"], "relevance_score": r["relevanceScore"],
             "text": text_sources[r["index"]]} for r in resp["results"]]


@strands_tool
def reranked_policy_retrieve(query: str) -> str:
    """Retrieve fraud policies from the Bedrock KB and rerank with Cohere 3.5."""
    resp = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )
    candidates = [i.get("content", {}).get("text", "")
                  for i in resp.get("retrievalResults", [])]
    if not candidates:
        return "No results found."
    ranked = _bedrock_rerank(query, candidates, num_results=3)
    return "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(ranked))


supervisor = Agent(
    model=llm,
    tools=[reranked_policy_retrieve, mem0_memory],
    system_prompt=(
        "You are a senior fraud investigator. For every question: "
        "1) Search your memory for prior findings. "
        "2) Retrieve the relevant fraud policies. "
        "3) Synthesize a clear compliance recommendation with policy citations. "
        "Always save your key findings to memory after answering."
    ),
    callback_handler=None,
)


def chat(message, history, user_id="investigator"):
    response = supervisor(message, user_id=user_id)
    return str(response)


gr.ChatInterface(
    fn=chat,
    title="Fraud Investigation Agent",
    description="Powered by Amazon Bedrock KB + Cohere Rerank 3.5 + mem0 memory.",
    additional_inputs=[
        gr.Textbox(value="investigator", label="Investigator ID (scopes memory)"),
    ],
).launch(server_name="0.0.0.0", server_port=7860, share=False)
