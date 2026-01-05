#!/usr/bin/env python3
"""Creative Leaps example for Living Memory Dynamics (LMD).

This example demonstrates the four creative leap operators:
1. Analogical Transfer - Cross-domain pattern transplant
2. Manifold Walker - Diffusion through embedding space
3. Orthogonal Composer - Gram-Schmidt perpendicular merges
4. Void Extrapolator - Ray-tracing into unexplored territory
"""

import torch
from lmd import (
    LivingMemory,
    ValenceTrajectory,
    NarrativePhase,
    CreativeLeapEngine,
    CreativeLeapConfig,
    LeapType,
    AnalogicalTransfer,
    ManifoldWalker,
    OrthogonalComposer,
    VoidExtrapolator,
)


def create_seed_memories(n_memories: int, content_dim: int) -> list:
    """Create seed memories for creative leaps."""
    memories = []
    for i in range(n_memories):
        embedding = torch.randn(content_dim)
        embedding = embedding / embedding.norm()

        memory = LivingMemory(
            id=f"seed_{i:03d}",
            content=embedding,
            energy=0.8,
            valence=ValenceTrajectory(points=torch.tensor([0.5, 0.7, 0.6])),
            phase=NarrativePhase.SETUP,
        )
        memories.append(memory)
    return memories


def demo_individual_operators(memories: list, content_dim: int):
    """Demonstrate each operator individually."""
    embeddings = [m.content for m in memories]

    print("\n" + "=" * 60)
    print("Individual Operator Demos")
    print("=" * 60)

    # 1. Analogical Transfer
    print("\n1. ANALOGICAL TRANSFER")
    print("   Transplants patterns between distant domains")
    print("   Like: 'dragon fire' + 'glass refraction' = 'prismatic breath'")

    analogical = AnalogicalTransfer(content_dim)
    leap = analogical.leap(embeddings, intensity=0.8)
    print(f"   Result: novelty={leap.novelty:.3f}, coherence={leap.coherence:.3f}")
    print(f"   Leap type: {leap.leap_type.name}")

    # 2. Manifold Walker
    print("\n2. MANIFOLD WALKER")
    print("   Diffuses through embedding space via multi-step noise")
    print("   Like: gradual concept morphing")

    walker = ManifoldWalker(content_dim, n_steps=10)
    leap = walker.leap(embeddings, intensity=0.7)
    print(f"   Result: novelty={leap.novelty:.3f}, coherence={leap.coherence:.3f}")
    print(f"   Steps taken: {walker.n_steps}")

    # 3. Orthogonal Composer
    print("\n3. ORTHOGONAL COMPOSER")
    print("   Uses Gram-Schmidt to find perpendicular concept axes")
    print("   Like: combining concepts along independent dimensions")

    composer = OrthogonalComposer(content_dim)
    leap = composer.leap(embeddings, intensity=0.9)
    print(f"   Result: novelty={leap.novelty:.3f}, coherence={leap.coherence:.3f}")
    print(f"   Orthogonal basis found: {len(embeddings)} vectors")

    # 4. Void Extrapolator
    print("\n4. VOID EXTRAPOLATOR")
    print("   Ray-traces into unexplored regions of embedding space")
    print("   Like: discovering genuinely novel concepts")

    extrapolator = VoidExtrapolator(content_dim, n_probes=50)
    leap = extrapolator.leap(embeddings, intensity=1.0)
    print(f"   Result: novelty={leap.novelty:.3f}, coherence={leap.coherence:.3f}")
    print(f"   Void density: low (unexplored territory)")


def demo_unified_engine(memories: list, content_dim: int):
    """Demonstrate the unified CreativeLeapEngine."""
    print("\n" + "=" * 60)
    print("Unified Creative Leap Engine")
    print("=" * 60)

    config = CreativeLeapConfig(
        content_dim=content_dim,
        analogical_weight=1.0,
        diffusion_weight=1.0,
        orthogonal_weight=1.0,
        extrapolation_weight=1.0,
    )
    engine = CreativeLeapEngine(config)

    embeddings = [m.content for m in memories]

    # Test different dopamine levels
    for dopamine in [0.3, 0.6, 0.9]:
        print(f"\n--- Dopamine Level: {dopamine} ---")
        print("   (Higher = more radical operators)")

        leaps = engine.batch_leap(embeddings, n_leaps=5, dopamine=dopamine)

        # Count leap types
        type_counts = {}
        for leap in leaps:
            name = leap.leap_type.name
            type_counts[name] = type_counts.get(name, 0) + 1

        print(f"   Generated {len(leaps)} leaps")
        print(f"   Types: {type_counts}")

        # Show best leap
        best = max(leaps, key=lambda l: l.novelty * l.coherence)
        print(f"   Best: {best.leap_type.name} "
              f"(novelty={best.novelty:.2f}, coherence={best.coherence:.2f})")


def demo_dopamine_modulation():
    """Show how dopamine affects operator selection."""
    print("\n" + "=" * 60)
    print("Dopamine Modulation of Creativity")
    print("=" * 60)

    print("""
   Low Dopamine (0.0-0.4):
   - Conservative operators
   - Manifold walking (gradual)
   - Safer, more coherent ideas

   Medium Dopamine (0.4-0.7):
   - Balanced exploration
   - Mix of all operators
   - Good novelty/coherence tradeoff

   High Dopamine (0.7-1.0):
   - Radical operators
   - Analogical transfer
   - Orthogonal composition
   - Higher novelty, may sacrifice coherence
    """)


def main():
    print("Living Memory Dynamics - Creative Leaps Example")
    print("=" * 60)

    # Configuration
    content_dim = 256
    n_memories = 15

    # Create seed memories
    print(f"\nCreating {n_memories} seed memories...")
    memories = create_seed_memories(n_memories, content_dim)
    print(f"Done. Each memory has {content_dim}-dim embedding.")

    # Demo individual operators
    demo_individual_operators(memories, content_dim)

    # Demo unified engine
    demo_unified_engine(memories, content_dim)

    # Explain dopamine modulation
    demo_dopamine_modulation()

    print("\n" + "=" * 60)
    print("Creative Leaps Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
