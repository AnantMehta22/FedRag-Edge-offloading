"""
=============================================================
  Real-Time Industry Use Case Simulator
  Multi-Agent RAG + FL Pipeline for Computation Offloading
  
  File: use_case_simulator.py
  Directory: c:\\Users\\admin\\Desktop\\RAG PIPELINE
  
  Simulates 5 real-world 5G/6G industry use cases:
    1. Autonomous Vehicles (V2X) — URLLC < 20ms
    2. Mobile AR/VR (Spatial Computing) — High Downlink
    3. Smart Factory IoT (Industry 4.0) — Multi-site FL
    4. Telemedicine (Remote Surgery) — Ultra-reliable
    5. Smart City Surveillance — Video Analytics
  
  Each use case generates a realistic synthetic task request,
  runs it through the multi-agent pipeline, and displays
  the grounded decision with full explanation.
=============================================================
"""

import os
import sys
import time
import random
import textwrap

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import from our core engine
from multi_agent_rag_fl import (
    TaskRequest, OffloadingResult, MultiAgentRAGFLPipeline
)


# ══════════════════════════════════════════════════════════════
# USE CASE DEFINITIONS
# ══════════════════════════════════════════════════════════════
USE_CASES = [
    {
        "id": 1,
        "icon": "🚗",
        "title": "Autonomous Vehicles & Connected Driving (V2X)",
        "scenario": (
            "A connected vehicle driving through cell tower CI=11803 encounters "
            "an obstacle and needs to execute real-time 3D point-cloud object "
            "detection for autonomous emergency braking."
        ),
        "deadline_ms": 20.0,
        "requirements": "Ultra-Reliable Low-Latency Communication (URLLC) < 20 ms",
        "task_params": {
            "cell_identity": 11803,
            "lac": 3288,
            "rat": 2,              # 5G NR
            "downlink_traffic": 8500,   # 8.5 MB point cloud return
            "uplink_traffic": 2200,     # 2.2 MB sensor data upload
            "duration": 15,             # 15ms compute requirement
            "latency": 3.5,             # Current observed latency
            "throughput": 850000,       # 850 Mbps 5G throughput
        },
        "rag_context": (
            "RAG queries historical V2X offloading records for CI=11803 under "
            "5G NR conditions. If edge cloud latency is guaranteed < 12ms, "
            "offloads point-cloud processing. If throughput drops, RAG redirects "
            "processing to the vehicle's onboard AI chip."
        ),
    },
    {
        "id": 2,
        "icon": "🥽",
        "title": "Mobile Augmented Reality (AR/VR) — Spatial Computing",
        "scenario": (
            "Users wearing AR glasses in a shopping mall rendering 3D digital "
            "overlays for product visualization. High-polygon 3D frame rendering "
            "requires massive downlink return data."
        ),
        "deadline_ms": 50.0,
        "requirements": "High Downlink (~30 MB per frame), Duration-sensitive rendering",
        "task_params": {
            "cell_identity": 24821,
            "lac": 3287,
            "rat": 2,              # 5G NR
            "downlink_traffic": 31000,  # ~30 MB rendered frame
            "uplink_traffic": 450,      # Camera pose data
            "duration": 40,             # 40ms render time
            "latency": 5.0,
            "throughput": 1099823,      # Max 5G throughput
        },
        "rag_context": (
            "RAG retrieves past rendering performance logs for this mall section "
            "(LAC=3287). Offloads high-polygon rendering to MEC server while "
            "keeping low-latency head-tracking computation on the glasses."
        ),
    },
    {
        "id": 3,
        "icon": "🏭",
        "title": "Smart Factory & Industrial IoT (Industry 4.0)",
        "scenario": (
            "Robotic arms on a factory floor inspecting products for defects "
            "via computer vision. Multiple factories across cities participate "
            "in federated learning without sharing proprietary video data."
        ),
        "deadline_ms": 100.0,
        "requirements": "Multi-site FL, network fluctuation handling, data privacy",
        "task_params": {
            "cell_identity": 7483,
            "lac": 3287,
            "rat": 1,              # 4G LTE (factory floor)
            "downlink_traffic": 900,    # Inspection result
            "uplink_traffic": 5500,     # High-res camera upload
            "duration": 82,             # Full inspection cycle
            "latency": 25.0,            # Moderate latency
            "throughput": 45000,        # Typical 4G industrial
        },
        "rag_context": (
            "FL allows multiple smart factories to train a joint offloading "
            "policy without sharing sensitive manufacturing data. Local RAG "
            "agents handle real-time factory floor network fluctuations."
        ),
    },
    {
        "id": 4,
        "icon": "🏥",
        "title": "Telemedicine & Remote Robotic Surgery",
        "scenario": (
            "A remote surgeon is performing a delicate procedure using robotic "
            "surgery equipment. The system requires absolute minimum latency "
            "with zero tolerance for packet loss or SLA violations."
        ),
        "deadline_ms": 10.0,
        "requirements": "Ultra-reliable, surgery-critical: Zero SLA violations allowed",
        "task_params": {
            "cell_identity": 40178,
            "lac": 43023,
            "rat": 2,              # 5G NR (dedicated slice)
            "downlink_traffic": 3200,   # Haptic feedback + video
            "uplink_traffic": 1800,     # Surgical instrument data
            "duration": 5,              # 5ms real-time control loop
            "latency": 1.0,             # Ultra-low baseline
            "throughput": 950000,       # Dedicated 5G slice
        },
        "rag_context": (
            "RAG enforces zero-tolerance SLA policy. Even minor network "
            "degradation triggers immediate fallback to local device processing "
            "for the control loop while video continues on edge."
        ),
    },
    {
        "id": 5,
        "icon": "🌆",
        "title": "Smart Cities & Intelligent Video Surveillance",
        "scenario": (
            "A network of IP cameras across a smart city performing real-time "
            "video analytics — crowd density estimation, anomaly detection, "
            "and license plate recognition. Batch processing across multiple "
            "camera feeds with varying network conditions."
        ),
        "deadline_ms": 200.0,
        "requirements": "Batch video analytics, varying network conditions per camera",
        "task_params": {
            "cell_identity": 20723,
            "lac": 43023,
            "rat": 1,              # 4G LTE
            "downlink_traffic": 150,    # Analytics results
            "uplink_traffic": 12000,    # Video stream upload
            "duration": 120,            # Batch processing
            "latency": 15.0,
            "throughput": 20000,        # Standard 4G
        },
        "rag_context": (
            "RAG checks historical logs for camera feeds in LAC=43023. "
            "Distributes analytics processing: crowd density on edge, "
            "license plate OCR locally for privacy compliance."
        ),
    },
]


# ══════════════════════════════════════════════════════════════
# SIMULATOR
# ══════════════════════════════════════════════════════════════
class UseCaseSimulator:
    """
    Simulates real-world 5G/6G industry use cases through
    the Multi-Agent RAG + FL pipeline.
    """
    
    def __init__(self, pipeline: MultiAgentRAGFLPipeline):
        self.pipeline = pipeline
        self.results = []
    
    def simulate_use_case(self, use_case: dict, verbose: bool = True) -> OffloadingResult:
        """Run a single use case through the pipeline."""
        
        if verbose:
            print(f"\n{'━' * 70}")
            print(f"  {use_case['icon']}  USE CASE {use_case['id']}: {use_case['title']}")
            print(f"{'━' * 70}")
            print(f"\n  📋 Scenario:")
            for line in textwrap.wrap(use_case['scenario'], width=64):
                print(f"     {line}")
            print(f"\n  ⏱️  Deadline: {use_case['deadline_ms']} ms")
            print(f"  📌 Requirements: {use_case['requirements']}")
        
        # Create task request
        params = use_case['task_params']
        task = TaskRequest(**params)
        
        if verbose:
            print(f"\n  📡 Task Request ({task.request_id}):")
            print(f"     Cell Identity : CI = {task.cell_identity}")
            print(f"     Location Area : LAC = {task.lac}")
            rat_name = "5G NR" if task.rat == 2 else "4G LTE"
            print(f"     Radio Access  : RAT = {task.rat} ({rat_name})")
            print(f"     Downlink      : {task.downlink_traffic:,.0f} KB")
            print(f"     Uplink        : {task.uplink_traffic:,.0f} KB")
            print(f"     Duration      : {task.duration} ms")
            print(f"     Latency       : {task.latency} ms")
            print(f"     Throughput    : {task.throughput:,.0f} Kbps")
        
        # Process through pipeline
        if verbose:
            print(f"\n  ⚙️  Processing through Multi-Agent Pipeline...")
        
        result = self.pipeline.process_task(task, verbose=verbose)
        self.results.append((use_case, result))
        
        # Display results
        if verbose:
            self._display_result(use_case, result)
        
        return result
    
    def _display_result(self, use_case: dict, result: OffloadingResult):
        """Display the three outputs for a use case."""
        
        # Output 1: Binary Offloading Decision
        decision_icon = "☁️" if result.decision == "Edge Cloud" else "📱"
        print(f"\n  {'─' * 60}")
        print(f"  OUTPUT 1: OFFLOADING DECISION")
        print(f"  {'─' * 60}")
        print(f"  {decision_icon}  Decision: {result.decision} (y={result.decision_code})")
        print(f"     Confidence: {result.confidence:.2%}")
        if result.override_applied:
            print(f"     ⚠️  OVERRIDE APPLIED: {result.override_reason[:80]}...")
        
        # Output 2: QoS Performance Metrics
        deadline = use_case['deadline_ms']
        latency_status = "✅ WITHIN" if result.predicted_latency_ms <= deadline else "❌ EXCEEDS"
        
        print(f"\n  {'─' * 60}")
        print(f"  OUTPUT 2: QoS PERFORMANCE METRICS")
        print(f"  {'─' * 60}")
        print(f"     Predicted Latency : {result.predicted_latency_ms:.2f} ms "
              f"{latency_status} deadline ({deadline} ms)")
        print(f"     Energy Consumption: {result.energy_consumption_j:.4f} J")
        print(f"     SLA Violation Prob: {result.sla_violation_probability:.2%}")
        
        # SLA bar visualization
        sla_bar_len = 30
        sla_fill = int(result.sla_violation_probability * sla_bar_len)
        sla_bar = "█" * sla_fill + "░" * (sla_bar_len - sla_fill)
        sla_color = "LOW" if result.sla_violation_probability < 0.1 else \
                    "MED" if result.sla_violation_probability < 0.5 else "HIGH"
        print(f"     SLA Risk: [{sla_bar}] {sla_color}")
        
        # Output 3: Grounded Explanation
        print(f"\n  {'─' * 60}")
        print(f"  OUTPUT 3: GROUNDED EXPLANATION")
        print(f"  {'─' * 60}")
        for line in textwrap.wrap(result.explanation, width=62):
            print(f"     {line}")
        
        if result.retrieved_citations:
            print(f"\n     📚 Retrieved Citations:")
            for citation in result.retrieved_citations:
                print(f"        {citation}")
        
        # RAG Context note
        print(f"\n     🔍 RAG Context:")
        for line in textwrap.wrap(use_case['rag_context'], width=60):
            print(f"        {line}")
        
        print(f"\n     ⏱️  Pipeline Processing Time: {result.processing_time_ms:.1f} ms")
    
    def simulate_all(self, verbose: bool = True) -> list:
        """Run all 5 use cases."""
        
        print("\n" + "=" * 70)
        print("  🌐 REAL-TIME INDUSTRY USE CASE SIMULATOR")
        print("  Multi-Agent RAG + FL Pipeline for 5G/6G Networks")
        print("=" * 70)
        
        all_results = []
        for use_case in USE_CASES:
            result = self.simulate_use_case(use_case, verbose=verbose)
            all_results.append(result)
        
        # Print summary table
        if verbose:
            self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: list):
        """Print a summary table of all use case results."""
        
        print(f"\n\n{'=' * 70}")
        print(f"  📊 USE CASE SIMULATION SUMMARY")
        print(f"{'=' * 70}")
        
        print(f"\n  {'Use Case':<25} {'Decision':<15} {'Latency':>10} {'Energy':>10} {'SLA Risk':>10}")
        print(f"  {'─' * 70}")
        
        for uc, result in zip(USE_CASES, results):
            short_title = uc['title'][:22] + "..." if len(uc['title']) > 25 else uc['title']
            icon = "☁️" if result.decision == "Edge Cloud" else "📱"
            sla = "LOW" if result.sla_violation_probability < 0.1 else \
                  "MED" if result.sla_violation_probability < 0.5 else "HIGH"
            
            print(f"  {uc['icon']} {short_title:<23} {icon} {result.decision:<12} "
                  f"{result.predicted_latency_ms:>8.1f}ms "
                  f"{result.energy_consumption_j:>8.4f}J "
                  f"{sla:>8}")
        
        # Aggregate stats
        edge_count = sum(1 for r in results if r.decision == "Edge Cloud")
        mobile_count = len(results) - edge_count
        avg_latency = sum(r.predicted_latency_ms for r in results) / len(results)
        avg_energy = sum(r.energy_consumption_j for r in results) / len(results)
        overrides = sum(1 for r in results if r.override_applied)
        
        print(f"\n  {'─' * 70}")
        print(f"  Total Use Cases    : {len(results)}")
        print(f"  Edge Cloud         : {edge_count}")
        print(f"  Mobile Device      : {mobile_count}")
        print(f"  Avg Latency        : {avg_latency:.2f} ms")
        print(f"  Avg Energy         : {avg_energy:.4f} J")
        print(f"  Guardrail Overrides: {overrides}")
        print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  Use Case Simulator — Standalone Mode")
    print("  Run via 'python run_offloading_demo.py' for full pipeline")
    print("=" * 70)
    print("\n  This module requires the full pipeline to be initialized.")
    print("  Execute run_offloading_demo.py for the complete experience.")
