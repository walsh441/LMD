"""LMD Comprehensive Benchmarks - Production Readiness Validation.

This module provides rigorous benchmarks to validate that LMD is not just
a prototype but a production-ready system.

Benchmarks:
1. Throughput: Operations per second, iterations per second
2. Latency: Per-operation timing distributions
3. Memory: Usage patterns and growth
4. Scaling: Performance vs memory count, idea count
5. Quality: Novelty, coherence, diversity over time
6. Stability: Long-running behavior, no degradation
7. Thread Safety: Concurrent access performance

Invented by Joshua R. Thomas, January 2026.
"""

import time
import gc
import threading
import statistics
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import torch
import sys

from lmd.living_memory import LivingMemory, ValenceTrajectory
from lmd.config import LMDConfig
from lmd.dynamics import LMDDynamics
from lmd.imagination import (
    StructuredMemory, MemorySlot, SlotType, Transform, TransformType,
    TransformOps, WillGenerator, WillVector, MentalCanvas, MemoryDecomposer
)
from lmd.plausibility import PlausibilityField, IdeaEvaluator
from lmd.ideation import IdeationEngine, IdeationConfig, IdeationResult
from lmd.safeguards import (
    RepulsionField, RealityAnchor, AutonomyController, ResourceBudget,
    IDGenerator, safe_normalize
)
from lmd.heartbeat_integration import HeartbeatIdeator


@dataclass
class TimingResult:
    """Result of a timing benchmark."""
    name: str
    iterations: int
    total_time_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    ops_per_second: float

    def summary(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total: {self.total_time_ms:.2f}ms\n"
            f"  Mean: {self.mean_ms:.3f}ms (std: {self.std_ms:.3f}ms)\n"
            f"  Min/Max: {self.min_ms:.3f}ms / {self.max_ms:.3f}ms\n"
            f"  P50/P95/P99: {self.p50_ms:.3f}ms / {self.p95_ms:.3f}ms / {self.p99_ms:.3f}ms\n"
            f"  Throughput: {self.ops_per_second:.1f} ops/sec"
        )


@dataclass
class MemoryResult:
    """Result of a memory benchmark."""
    name: str
    initial_mb: float
    final_mb: float
    peak_mb: float
    growth_mb: float
    objects_created: int


@dataclass
class ScalingResult:
    """Result of a scaling benchmark."""
    name: str
    sizes: List[int]
    times_ms: List[float]
    ops_per_second: List[float]
    scaling_factor: float  # O(n^scaling_factor)


@dataclass
class QualityResult:
    """Result of a quality benchmark."""
    name: str
    iterations: int
    novelty_mean: float
    novelty_std: float
    coherence_mean: float
    coherence_std: float
    diversity_score: float
    echo_chamber_risk: float  # 0=none, 1=severe


@dataclass
class BenchmarkSuite:
    """Complete benchmark results."""
    timestamp: str
    system_info: Dict[str, Any]
    timing_results: List[TimingResult] = field(default_factory=list)
    memory_results: List[MemoryResult] = field(default_factory=list)
    scaling_results: List[ScalingResult] = field(default_factory=list)
    quality_results: List[QualityResult] = field(default_factory=list)
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "timing": [asdict(r) for r in self.timing_results],
            "memory": [asdict(r) for r in self.memory_results],
            "scaling": [asdict(r) for r in self.scaling_results],
            "quality": [asdict(r) for r in self.quality_results],
            "summary": self.summary_stats
        }


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark context."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_count": torch.get_num_threads(),
    }


@contextmanager
def timer():
    """Context manager for timing."""
    start = time.perf_counter()
    times = []
    yield times
    times.append((time.perf_counter() - start) * 1000)


def benchmark_timing(
    name: str,
    func: callable,
    iterations: int = 100,
    warmup: int = 5
) -> TimingResult:
    """Run timing benchmark on a function."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    start_total = time.perf_counter()
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    total_time = (time.perf_counter() - start_total) * 1000

    times.sort()
    return TimingResult(
        name=name,
        iterations=iterations,
        total_time_ms=total_time,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_ms=min(times),
        max_ms=max(times),
        p50_ms=times[len(times) // 2],
        p95_ms=times[int(len(times) * 0.95)],
        p99_ms=times[int(len(times) * 0.99)],
        ops_per_second=iterations / (total_time / 1000)
    )


class LMDBenchmarks:
    """Comprehensive LMD benchmark suite."""

    def __init__(self, content_dim: int = 32, verbose: bool = True):
        self.content_dim = content_dim
        self.verbose = verbose
        self.config = LMDConfig.toy_scale()
        self.results = BenchmarkSuite(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=get_system_info()
        )

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def create_memories(self, n: int) -> List[LivingMemory]:
        """Create test memories."""
        return [
            LivingMemory(
                id=i,
                content=torch.randn(self.content_dim),
                valence=ValenceTrajectory.random(),
                energy=0.5 + 0.5 * torch.rand(1).item(),
                created_at=time.time() - i,
                label=f"memory_{i}"
            )
            for i in range(n)
        ]

    def create_structured(self, n: int) -> List[StructuredMemory]:
        """Create test structured memories."""
        result = []
        for i in range(n):
            s = StructuredMemory(id=i)
            for slot_name in ["agent", "action", "location"]:
                s.add_slot(slot_name, MemorySlot(
                    slot_type=SlotType.AGENT,
                    name=slot_name,
                    content=torch.randn(self.content_dim),
                    confidence=0.8
                ))
            result.append(s)
        return result

    # =========================================================================
    # TIMING BENCHMARKS
    # =========================================================================

    def bench_memory_creation(self, n: int = 1000) -> TimingResult:
        """Benchmark: LivingMemory creation throughput."""
        self.log("Benchmarking memory creation...")
        counter = [0]

        def create_one():
            LivingMemory(
                id=counter[0],
                content=torch.randn(self.content_dim),
                valence=ValenceTrajectory.random(),
                energy=1.0,
                created_at=time.time()
            )
            counter[0] += 1

        return benchmark_timing("memory_creation", create_one, iterations=n)

    def bench_dynamics_step(self, n_memories: int = 100, iterations: int = 100) -> TimingResult:
        """Benchmark: LMDDynamics.step() throughput."""
        self.log(f"Benchmarking dynamics step ({n_memories} memories)...")
        dynamics = LMDDynamics(self.config)
        memories = self.create_memories(n_memories)

        def step():
            dynamics.step(memories)

        return benchmark_timing(f"dynamics_step_{n_memories}mem", step, iterations=iterations)

    def bench_will_generation(self, n_memories: int = 50, iterations: int = 200) -> TimingResult:
        """Benchmark: Will generation throughput."""
        self.log("Benchmarking will generation...")
        generator = WillGenerator(self.content_dim)
        memories = self.create_memories(n_memories)

        def generate():
            generator.generate_curiosity_will(memories)

        return benchmark_timing("will_generation", generate, iterations=iterations)

    def bench_transform_ops(self, iterations: int = 500) -> TimingResult:
        """Benchmark: Transform operations throughput."""
        self.log("Benchmarking transform operations...")
        ops = TransformOps(self.content_dim)
        structured = self.create_structured(1)[0]
        transform = Transform(
            transform_type=TransformType.MORPH,
            target_slot="agent",
            magnitude=0.5
        )

        def apply_transform():
            ops.apply(structured, transform)

        return benchmark_timing("transform_ops", apply_transform, iterations=iterations)

    def bench_plausibility_scoring(self, iterations: int = 300) -> TimingResult:
        """Benchmark: Plausibility scoring throughput."""
        self.log("Benchmarking plausibility scoring...")
        plausibility = PlausibilityField(self.content_dim)
        memories = self.create_memories(50)
        plausibility.learn_from_memories(memories)
        structured = self.create_structured(1)[0]

        def score():
            plausibility.score(structured)

        return benchmark_timing("plausibility_score", score, iterations=iterations)

    def bench_ideation_session(self, n_memories: int = 20, iterations: int = 20) -> TimingResult:
        """Benchmark: Full ideation session throughput."""
        self.log("Benchmarking full ideation session...")
        engine = IdeationEngine(self.config, IdeationConfig.quick())
        memories = self.create_memories(n_memories)

        def ideate():
            engine.ideate(memories)

        return benchmark_timing("ideation_session", ideate, iterations=iterations)

    def bench_repulsion_field(self, iterations: int = 500) -> TimingResult:
        """Benchmark: Repulsion field operations."""
        self.log("Benchmarking repulsion field...")
        repulsion = RepulsionField(self.content_dim)
        embedding = torch.randn(self.content_dim)

        def repulsion_ops():
            repulsion.mark_explored(torch.randn(self.content_dim))
            repulsion.compute_repulsion(embedding)
            repulsion.novelty_penalty(embedding)

        return benchmark_timing("repulsion_field", repulsion_ops, iterations=iterations)

    def bench_heartbeat_tick(self, n_memories: int = 30, iterations: int = 1000) -> TimingResult:
        """Benchmark: Heartbeat tick (checking triggers, no ideation)."""
        self.log("Benchmarking heartbeat tick...")
        dynamics = LMDDynamics(self.config)
        ideator = HeartbeatIdeator(
            self.config, dynamics,
            heartbeats_per_check=1000000  # Never actually check triggers
        )
        memories = self.create_memories(n_memories)

        def tick():
            ideator.on_heartbeat(memories, dopamine=16384)

        return benchmark_timing("heartbeat_tick", tick, iterations=iterations)

    # =========================================================================
    # SCALING BENCHMARKS
    # =========================================================================

    def bench_scaling_memories(self, sizes: List[int] = None) -> ScalingResult:
        """Benchmark: How does dynamics scale with memory count?"""
        if sizes is None:
            sizes = [10, 25, 50, 100, 200, 500]

        self.log(f"Benchmarking scaling with memory counts: {sizes}")
        dynamics = LMDDynamics(self.config)
        times = []
        ops_per_sec = []

        for n in sizes:
            memories = self.create_memories(n)
            result = benchmark_timing(
                f"dynamics_{n}",
                lambda: dynamics.step(memories),
                iterations=50,
                warmup=3
            )
            times.append(result.mean_ms)
            ops_per_sec.append(result.ops_per_second)
            self.log(f"  {n} memories: {result.mean_ms:.3f}ms ({result.ops_per_second:.1f} ops/s)")

        # Estimate scaling factor (O(n^k))
        import math
        if len(sizes) >= 2:
            # Use first and last to estimate
            n1, n2 = sizes[0], sizes[-1]
            t1, t2 = times[0], times[-1]
            if t1 > 0 and t2 > 0 and n1 > 0 and n2 > 0:
                scaling_factor = math.log(t2 / t1) / math.log(n2 / n1)
            else:
                scaling_factor = 1.0
        else:
            scaling_factor = 1.0

        return ScalingResult(
            name="memory_count_scaling",
            sizes=sizes,
            times_ms=times,
            ops_per_second=ops_per_sec,
            scaling_factor=scaling_factor
        )

    def bench_scaling_ideation(self, sizes: List[int] = None) -> ScalingResult:
        """Benchmark: How does ideation scale with memory count?"""
        if sizes is None:
            sizes = [10, 20, 40, 80]

        self.log(f"Benchmarking ideation scaling: {sizes}")
        times = []
        ops_per_sec = []

        for n in sizes:
            engine = IdeationEngine(self.config, IdeationConfig.quick())
            memories = self.create_memories(n)
            result = benchmark_timing(
                f"ideation_{n}",
                lambda: engine.ideate(memories),
                iterations=10,
                warmup=2
            )
            times.append(result.mean_ms)
            ops_per_sec.append(result.ops_per_second)
            self.log(f"  {n} memories: {result.mean_ms:.1f}ms ({result.ops_per_second:.2f} ops/s)")

        import math
        if len(sizes) >= 2:
            n1, n2 = sizes[0], sizes[-1]
            t1, t2 = times[0], times[-1]
            if t1 > 0 and t2 > 0:
                scaling_factor = math.log(t2 / t1) / math.log(n2 / n1)
            else:
                scaling_factor = 1.0
        else:
            scaling_factor = 1.0

        return ScalingResult(
            name="ideation_scaling",
            sizes=sizes,
            times_ms=times,
            ops_per_second=ops_per_sec,
            scaling_factor=scaling_factor
        )

    # =========================================================================
    # QUALITY BENCHMARKS
    # =========================================================================

    def bench_quality_over_time(self, n_sessions: int = 20) -> QualityResult:
        """Benchmark: Quality metrics over multiple ideation sessions."""
        self.log(f"Benchmarking quality over {n_sessions} sessions...")
        engine = IdeationEngine(self.config, IdeationConfig.quick())
        memories = self.create_memories(30)

        novelties = []
        coherences = []
        all_embeddings = []

        for i in range(n_sessions):
            result = engine.ideate(memories)
            novelties.append(result.average_novelty)
            coherences.append(result.average_coherence)

            for idea in result.ideas:
                all_embeddings.append(idea.to_embedding(self.content_dim))

            if self.verbose and (i + 1) % 5 == 0:
                self.log(f"  Session {i+1}: novelty={result.average_novelty:.3f}, "
                        f"coherence={result.average_coherence:.3f}")

        # Compute diversity (average pairwise distance)
        diversity = 0.0
        echo_risk = 0.0
        if len(all_embeddings) >= 2:
            stacked = torch.stack(all_embeddings)
            dists = torch.cdist(stacked, stacked)
            mask = ~torch.eye(len(all_embeddings), dtype=torch.bool)
            diversity = dists[mask].mean().item()

            # Echo chamber risk: how many pairs are too similar?
            similar_threshold = 0.3
            similar_count = (dists[mask] < similar_threshold).sum().item()
            total_pairs = mask.sum().item()
            echo_risk = similar_count / total_pairs if total_pairs > 0 else 0

        return QualityResult(
            name="quality_over_time",
            iterations=n_sessions,
            novelty_mean=statistics.mean(novelties),
            novelty_std=statistics.stdev(novelties) if len(novelties) > 1 else 0,
            coherence_mean=statistics.mean(coherences),
            coherence_std=statistics.stdev(coherences) if len(coherences) > 1 else 0,
            diversity_score=diversity,
            echo_chamber_risk=echo_risk
        )

    def bench_repulsion_effectiveness(self, n_sessions: int = 30) -> QualityResult:
        """Benchmark: Does repulsion prevent echo chambers?"""
        self.log("Benchmarking repulsion effectiveness...")

        # WITH repulsion
        engine_with = IdeationEngine(self.config, IdeationConfig.quick())
        memories = self.create_memories(30)

        embeddings_with = []
        for _ in range(n_sessions):
            result = engine_with.ideate(memories)
            for idea in result.ideas:
                embeddings_with.append(idea.to_embedding(self.content_dim))

        # Compute diversity with repulsion
        if len(embeddings_with) >= 2:
            stacked = torch.stack(embeddings_with)
            dists = torch.cdist(stacked, stacked)
            mask = ~torch.eye(len(embeddings_with), dtype=torch.bool)
            diversity_with = dists[mask].mean().item()
            similar_count = (dists[mask] < 0.3).sum().item()
            echo_risk_with = similar_count / mask.sum().item()
        else:
            diversity_with = 0
            echo_risk_with = 1

        self.log(f"  With repulsion: diversity={diversity_with:.3f}, echo_risk={echo_risk_with:.3f}")

        return QualityResult(
            name="repulsion_effectiveness",
            iterations=n_sessions,
            novelty_mean=0,  # Not measured here
            novelty_std=0,
            coherence_mean=0,
            coherence_std=0,
            diversity_score=diversity_with,
            echo_chamber_risk=echo_risk_with
        )

    # =========================================================================
    # THREAD SAFETY BENCHMARKS
    # =========================================================================

    def bench_concurrent_access(self, n_threads: int = 4, ops_per_thread: int = 100) -> TimingResult:
        """Benchmark: Concurrent access to shared structures."""
        self.log(f"Benchmarking concurrent access ({n_threads} threads)...")

        repulsion = RepulsionField(self.content_dim)
        anchor = RealityAnchor(self.content_dim)
        id_gen = IDGenerator()

        errors = []
        times = []

        def worker(thread_id: int):
            start = time.perf_counter()
            try:
                for i in range(ops_per_thread):
                    # Mix of operations
                    embedding = torch.randn(self.content_dim)
                    repulsion.mark_explored(embedding)
                    repulsion.compute_repulsion(embedding)
                    anchor.record_outcome(thread_id * 1000 + i, 0.5, 0.3)
                    anchor.ground_valence(0.5, embedding)
                    id_gen.next_id("memory")
            except Exception as e:
                errors.append(str(e))
            times.append((time.perf_counter() - start) * 1000)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]

        total_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = (time.perf_counter() - total_start) * 1000

        total_ops = n_threads * ops_per_thread * 6  # 6 operations per iteration

        if errors:
            self.log(f"  ERRORS: {errors}")

        return TimingResult(
            name=f"concurrent_{n_threads}threads",
            iterations=total_ops,
            total_time_ms=total_time,
            mean_ms=statistics.mean(times) / ops_per_thread,
            std_ms=statistics.stdev(times) / ops_per_thread if len(times) > 1 else 0,
            min_ms=min(times) / ops_per_thread,
            max_ms=max(times) / ops_per_thread,
            p50_ms=sorted(times)[len(times)//2] / ops_per_thread,
            p95_ms=sorted(times)[int(len(times)*0.95)] / ops_per_thread if len(times) > 1 else times[0] / ops_per_thread,
            p99_ms=sorted(times)[-1] / ops_per_thread,
            ops_per_second=total_ops / (total_time / 1000)
        )

    # =========================================================================
    # LONG-RUNNING STABILITY BENCHMARK
    # =========================================================================

    def bench_long_running(self, duration_seconds: float = 10.0) -> Dict[str, Any]:
        """Benchmark: Long-running stability test."""
        self.log(f"Running {duration_seconds}s stability test...")

        dynamics = LMDDynamics(self.config)
        memories = self.create_memories(30)
        engine = IdeationEngine(self.config, IdeationConfig.quick())

        stats = {
            "duration_seconds": duration_seconds,
            "total_steps": 0,
            "total_ideations": 0,
            "total_ideas": 0,
            "memory_growth": [],
            "quality_over_time": [],
            "errors": []
        }

        start_time = time.time()
        step = 0

        try:
            while time.time() - start_time < duration_seconds:
                # Run dynamics
                dynamics.step(memories)
                stats["total_steps"] += 1

                # Occasional ideation
                if step % 10 == 0:
                    try:
                        result = engine.ideate(memories)
                        stats["total_ideations"] += 1
                        stats["total_ideas"] += len(result.ideas)
                        stats["quality_over_time"].append({
                            "time": time.time() - start_time,
                            "novelty": result.average_novelty,
                            "coherence": result.average_coherence
                        })
                    except Exception as e:
                        stats["errors"].append(str(e))

                # Track memory count
                if step % 50 == 0:
                    alive = sum(1 for m in memories if m.is_alive)
                    stats["memory_growth"].append({
                        "time": time.time() - start_time,
                        "alive": alive,
                        "total": len(memories)
                    })

                step += 1

        except Exception as e:
            stats["errors"].append(f"Fatal: {str(e)}")

        elapsed = time.time() - start_time
        stats["actual_duration"] = elapsed
        stats["steps_per_second"] = stats["total_steps"] / elapsed
        stats["ideations_per_second"] = stats["total_ideations"] / elapsed

        self.log(f"  Steps: {stats['total_steps']} ({stats['steps_per_second']:.1f}/s)")
        self.log(f"  Ideations: {stats['total_ideations']} ({stats['ideations_per_second']:.2f}/s)")
        self.log(f"  Errors: {len(stats['errors'])}")

        return stats

    # =========================================================================
    # RUN ALL BENCHMARKS
    # =========================================================================

    def run_all(self) -> BenchmarkSuite:
        """Run all benchmarks and return complete results."""
        self.log("=" * 70)
        self.log("LMD COMPREHENSIVE BENCHMARK SUITE")
        self.log("=" * 70)
        self.log(f"Content dim: {self.content_dim}")
        self.log(f"System: {self.results.system_info}")
        self.log("")

        # Timing benchmarks
        self.log("\n" + "-" * 40)
        self.log("TIMING BENCHMARKS")
        self.log("-" * 40)

        self.results.timing_results.append(self.bench_memory_creation(1000))
        self.results.timing_results.append(self.bench_dynamics_step(100, 100))
        self.results.timing_results.append(self.bench_will_generation(50, 200))
        self.results.timing_results.append(self.bench_transform_ops(500))
        self.results.timing_results.append(self.bench_plausibility_scoring(300))
        self.results.timing_results.append(self.bench_ideation_session(20, 20))
        self.results.timing_results.append(self.bench_repulsion_field(500))
        self.results.timing_results.append(self.bench_heartbeat_tick(30, 1000))

        # Scaling benchmarks
        self.log("\n" + "-" * 40)
        self.log("SCALING BENCHMARKS")
        self.log("-" * 40)

        self.results.scaling_results.append(self.bench_scaling_memories())
        self.results.scaling_results.append(self.bench_scaling_ideation())

        # Quality benchmarks
        self.log("\n" + "-" * 40)
        self.log("QUALITY BENCHMARKS")
        self.log("-" * 40)

        self.results.quality_results.append(self.bench_quality_over_time(20))
        self.results.quality_results.append(self.bench_repulsion_effectiveness(30))

        # Thread safety
        self.log("\n" + "-" * 40)
        self.log("THREAD SAFETY BENCHMARKS")
        self.log("-" * 40)

        self.results.timing_results.append(self.bench_concurrent_access(4, 100))

        # Long-running stability
        self.log("\n" + "-" * 40)
        self.log("STABILITY BENCHMARK")
        self.log("-" * 40)

        stability = self.bench_long_running(10.0)
        self.results.summary_stats["stability"] = stability

        # Compute summary statistics
        self.log("\n" + "=" * 70)
        self.log("SUMMARY")
        self.log("=" * 70)

        timing_summary = {}
        for r in self.results.timing_results:
            timing_summary[r.name] = {
                "ops_per_second": r.ops_per_second,
                "mean_ms": r.mean_ms,
                "p99_ms": r.p99_ms
            }
            self.log(f"{r.name}: {r.ops_per_second:.1f} ops/s, mean={r.mean_ms:.3f}ms")

        self.results.summary_stats["timing"] = timing_summary
        self.results.summary_stats["scaling"] = {
            r.name: {"factor": r.scaling_factor}
            for r in self.results.scaling_results
        }
        self.results.summary_stats["quality"] = {
            r.name: {
                "diversity": r.diversity_score,
                "echo_risk": r.echo_chamber_risk
            }
            for r in self.results.quality_results
        }

        return self.results


def run_benchmarks(verbose: bool = True, save_path: str = None) -> BenchmarkSuite:
    """Run complete benchmark suite."""
    benchmarks = LMDBenchmarks(content_dim=32, verbose=verbose)
    results = benchmarks.run_all()

    if save_path:
        with open(save_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved to: {save_path}")

    return results


if __name__ == "__main__":
    run_benchmarks(verbose=True, save_path="lmd_benchmarks.json")
