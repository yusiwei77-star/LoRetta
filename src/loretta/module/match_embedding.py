"""Match-embedding construction used by LoRetta's affine locator."""

import torch
from einops import einsum


def _normalize_descriptors(descriptors: torch.Tensor) -> torch.Tensor:
    return descriptors / descriptors.norm(dim=-1, keepdim=True)


def build_match_embeddings(
    *,
    reference_descriptors: torch.Tensor,
    sensed_descriptors: torch.Tensor,
    sensed_position_embeddings: torch.Tensor,
    temperature: float,
    no_match_embedding: torch.Tensor,
    no_match_logit: torch.Tensor,
) -> torch.Tensor:
    batch_size, reference_height, reference_width, _ = reference_descriptors.shape
    _, sensed_height, sensed_width, _ = sensed_descriptors.shape
    reference_descriptors = _normalize_descriptors(reference_descriptors)
    sensed_descriptors = _normalize_descriptors(sensed_descriptors)
    similarities = einsum(
        reference_descriptors,
        sensed_descriptors,
        "B HR WR D, B HS WS D -> B HR WR HS WS",
    )
    matching_logits = (1 / temperature * similarities).reshape(
        batch_size,
        reference_height * reference_width,
        sensed_height * sensed_width,
    )
    matching_logits = torch.cat(
        [
            matching_logits,
            no_match_logit.to(matching_logits).expand(
                batch_size, reference_height * reference_width, 1
            ),
        ],
        dim=2,
    )
    match_probabilities = torch.softmax(matching_logits, dim=2)

    embedding_candidates = sensed_position_embeddings.reshape(
        batch_size, sensed_height * sensed_width, -1
    )
    no_match_embedding = no_match_embedding.to(embedding_candidates).reshape(1, 1, -1)
    embedding_candidates = torch.cat(
        [embedding_candidates, no_match_embedding.expand(batch_size, 1, -1)], dim=1
    )
    match_embeddings = einsum(
        match_probabilities,
        embedding_candidates,
        "B NR NS, B NS D -> B NR D",
    )
    return match_embeddings.reshape(batch_size, reference_height, reference_width, -1)
