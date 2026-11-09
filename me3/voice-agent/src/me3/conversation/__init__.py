"""Phase 3 conversation: turn-taking, Haiku/MockLLM, overlap, acks.

Orchestration lives in me3.session + me3.conversation (local audio stack).
pipecat-ai==1.15.0 remains pinned for later benchmark/frame integration;
Phase 3 does not route mic PCM through Pipecat transports.
"""

from me3.conversation.pipeline import ConversationOrchestrator

__all__ = ["ConversationOrchestrator"]
