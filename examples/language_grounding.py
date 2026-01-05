#!/usr/bin/env python3
"""Language Grounding example for Living Memory Dynamics (LMD).

This example demonstrates:
1. Encoding text to embeddings
2. Building a text corpus for retrieval
3. Decoding embeddings back to human-readable text
4. Describing creative leaps in natural language

Requirements:
    pip install living-memory-dynamics[language]
"""

import torch
from lmd import (
    LanguageGrounding,
    GroundingConfig,
    EncoderType,
    create_grounding,
    LivingMemory,
    ValenceTrajectory,
    NarrativePhase,
    CreativeLeapEngine,
    CreativeLeapConfig,
    SENTENCE_TRANSFORMERS_AVAILABLE,
)


def check_dependencies():
    """Check if required dependencies are installed."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("=" * 60)
        print("sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        print("Or: pip install living-memory-dynamics[language]")
        print("=" * 60)
        return False
    return True


def demo_basic_encoding():
    """Demonstrate basic text encoding."""
    print("\n" + "=" * 60)
    print("1. Basic Text Encoding")
    print("=" * 60)

    # Create grounding with MiniLM (fast, 384 dim)
    grounding = create_grounding(encoder="minilm")
    print(f"   Encoder: MiniLM")
    print(f"   Embedding dimension: {grounding.embedding_dim}")

    # Encode single text
    text = "a fire-breathing dragon"
    embedding = grounding.encode(text)
    print(f"\n   Text: '{text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm: {embedding.norm().item():.4f}")

    # Encode multiple texts
    texts = [
        "crystalline glass structure",
        "prismatic light refraction",
        "ancient mythical creature",
    ]
    embeddings = grounding.encode(texts)
    print(f"\n   Encoded {len(texts)} texts")
    print(f"   Batch shape: {embeddings.shape}")

    return grounding


def demo_corpus_building(grounding: LanguageGrounding):
    """Demonstrate building a text corpus."""
    print("\n" + "=" * 60)
    print("2. Building Text Corpus")
    print("=" * 60)

    # Add texts to corpus
    corpus_texts = [
        "a fire-breathing dragon",
        "crystalline glass structure",
        "prismatic light refraction",
        "ancient mythical creature",
        "rainbow spectrum of colors",
        "molten lava flow",
        "frozen ice sculpture",
        "diamond crystal lattice",
        "phoenix rising from flames",
        "aurora borealis lights",
        "volcanic eruption",
        "glacier formation",
        "sunlight through prism",
        "stained glass window",
        "dragon scales armor",
    ]

    print(f"   Adding {len(corpus_texts)} texts to corpus...")
    for text in corpus_texts:
        grounding.add_to_corpus(text)

    print(f"   Corpus size: {grounding.corpus_size}")

    return corpus_texts


def demo_retrieval(grounding: LanguageGrounding):
    """Demonstrate retrieval-based decoding."""
    print("\n" + "=" * 60)
    print("3. Retrieval-Based Decoding")
    print("=" * 60)

    # Create a query embedding (average of two concepts)
    dragon_emb = grounding.encode("a fire-breathing dragon")
    glass_emb = grounding.encode("crystalline glass structure")

    # Blend them (simulating a creative leap result)
    blended = (dragon_emb + glass_emb) / 2
    blended = blended / blended.norm()

    print("   Query: blend of 'dragon' and 'glass'")

    # Decode
    result = grounding.decode(blended, top_k=5)

    print(f"\n   Top matches:")
    for i, (text, similarity) in enumerate(result.matches, 1):
        print(f"      {i}. '{text}' (similarity: {similarity:.3f})")

    print(f"\n   Interpolated description: {result.interpolated_description}")
    print(f"   Confidence: {result.confidence:.3f}")


def demo_creative_leap_description(grounding: LanguageGrounding):
    """Demonstrate describing creative leaps."""
    print("\n" + "=" * 60)
    print("4. Creative Leap Descriptions")
    print("=" * 60)

    # Get source embeddings
    sources = [
        grounding.encode("a fire-breathing dragon"),
        grounding.encode("crystalline glass structure"),
    ]
    source_texts = ["fire-breathing dragon", "crystalline glass"]

    # Create a "leap" result (perpendicular composition)
    # This simulates what OrthogonalComposer might produce
    result = sources[0] + sources[1]
    result = result / result.norm()
    # Add some novelty
    noise = torch.randn_like(result) * 0.3
    result = result + noise
    result = result / result.norm()

    # Describe the leap
    description = grounding.describe_leap(
        leap_type="ORTHOGONAL",
        sources=sources,
        result=result,
        source_texts=source_texts
    )

    print(f"   Leap type: {description.leap_type}")
    print(f"   Sources: {description.source_texts}")
    print(f"   Novelty score: {description.novelty_score:.3f}")
    print(f"   Grounding confidence: {description.grounding_confidence:.3f}")
    print(f"\n   Synthesized description:")
    print(f"      {description.synthesized_description}")
    print(f"\n   Nearest concepts in corpus:")
    for text, sim in description.nearest_texts[:3]:
        print(f"      - '{text}' ({sim:.3f})")


def demo_memory_grounding(grounding: LanguageGrounding):
    """Demonstrate grounding LivingMemory objects."""
    print("\n" + "=" * 60)
    print("5. Grounding Living Memories")
    print("=" * 60)

    # Create memories with encoded content
    memories = []
    concepts = [
        "ancient dragon guarding treasure",
        "crystal cave with glowing gems",
        "sunset over mountain peaks",
    ]

    for i, concept in enumerate(concepts):
        embedding = grounding.encode(concept)
        memory = LivingMemory(
            id=f"memory_{i}",
            content=embedding,
            energy=0.8,
            valence=ValenceTrajectory(points=torch.tensor([0.5, 0.7, 0.6])),
            phase=NarrativePhase.SETUP,
        )
        memories.append(memory)
        # Also add to corpus
        grounding.add_to_corpus(concept)

    print(f"   Created {len(memories)} memories")

    # Ground memories back to text
    print("\n   Grounded memories:")
    grounded = grounding.ground_memories(memories)
    for g in grounded:
        print(f"      ID: {g.metadata['memory_id']}")
        print(f"      Text: '{g.text}'")
        print(f"      Confidence: {g.metadata['confidence']:.3f}")
        print()


def demo_with_creative_leaps(grounding: LanguageGrounding):
    """Demonstrate integration with CreativeLeapEngine."""
    print("\n" + "=" * 60)
    print("6. Integration with Creative Leaps")
    print("=" * 60)

    # Create leap engine matching grounding dimension
    config = CreativeLeapConfig(content_dim=grounding.embedding_dim)
    leap_engine = CreativeLeapEngine(config)

    # Encode source concepts (move to CPU for leap engine compatibility)
    sources = [
        grounding.encode("volcanic eruption").cpu(),
        grounding.encode("frozen ice sculpture").cpu(),
        grounding.encode("rainbow spectrum").cpu(),
    ]

    print("   Sources: volcanic eruption, frozen ice, rainbow spectrum")
    print("   Generating creative leaps...\n")

    # Generate leaps with different operators
    for dopamine in [0.3, 0.7, 0.95]:
        leap = leap_engine.leap(sources, dopamine=dopamine)

        # Describe the leap
        description = grounding.describe_leap(
            leap_type=leap.leap_type.name,
            sources=sources,
            result=leap.embedding,
        )

        print(f"   Dopamine: {dopamine}")
        print(f"   Leap type: {leap.leap_type.name}")
        print(f"   Description: {description.synthesized_description}")
        print(f"   Novelty: {description.novelty_score:.2f}")
        print()


def main():
    print("Living Memory Dynamics - Language Grounding Example")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        return

    # Run demos
    grounding = demo_basic_encoding()
    demo_corpus_building(grounding)
    demo_retrieval(grounding)
    demo_creative_leap_description(grounding)
    demo_memory_grounding(grounding)
    demo_with_creative_leaps(grounding)

    print("\n" + "=" * 60)
    print("Language Grounding Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
