import os
from RAG import StudyMaterialRAG
import sys
from material import generate_materials
from prompts import Topic

def main():
    rag = StudyMaterialRAG()

    syllabus = """
    Module 1: Cloud Computing Fundamentals
    Covers virtualization, hypervisors, resource pooling.
    Module 2: Containerization
    Includes Docker architecture, images, registries, orchestration.
    """
    # rag.add_syllabus(syllabus, {"course_id": "C1", "teacher_id": "T1"})

    # Add multiple reference documents to stress-test hybrid retrieval
    reference_corpus = [
        """
        Virtualization allows abstraction of physical hardware resources. Hypervisors (Type 1 bare‑metal and Type 2 hosted)
        manage guest operating systems. CPU scheduling, memory ballooning, and I/O device emulation enable consolidation.
        """,
        """
        Containerization differs from full virtualization: lightweight process isolation using kernel namespaces and cgroups.
        Docker image layering (AUFS/OverlayFS) supports efficient distribution. Registry caching accelerates CI/CD pipelines.
        """,
        """
        Kubernetes orchestrates containers: scheduler assigns Pods based on resource requests; controller manager maintains desired state;
        kube-proxy manages virtual IP based service discovery; etcd provides strongly consistent key-value store for cluster metadata.
        """,
        """
        Resource pooling in cloud computing aggregates compute, storage, and network bandwidth to improve utilization.
        Elastic scaling policies (target tracking, step scaling) respond to metrics like CPU, latency, or queue depth.
        """,
        """
        Network virtualization introduces overlay networks (VXLAN, Geneve) decoupling logical topology from physical fabric,
        enabling multi-tenant isolation. Virtual switches (OVS) enforce ACLs and QoS. Service mesh adds L7 traffic management.
        """,
        """
        Hypervisor security hardening includes micro-segmentation, secure boot chains, encrypted VM images, and runtime integrity checks.
        Side-channel mitigations (e.g., for speculative execution) can impact consolidation ratios and scheduling fairness.
        """,
        """
        Container security scanning inspects image layers for CVEs; SBOM (Software Bill of Materials) supports provenance.
        Runtime enforcement with seccomp, AppArmor, SELinux, and eBPF-based syscall profiling reduces attack surface.
        """,
        """
        Serverless (FaaS) builds atop container or microVM isolation (Firecracker). Cold start latency reduced via snapshot/restore
        and provisioned concurrency. Event-driven autoscaling differs from traditional request-per-second based triggers.
        """,
        """
        Virtualization performance tuning: NUMA-aware placement, huge pages for TLB efficiency, SR-IOV for near line-rate networking,
        paravirtualized drivers (virtio) reduce emulation overhead, and CPU pinning stabilizes latency-sensitive workloads.
        """,
        """
        Advanced orchestration: topology-aware scheduling leverages zone / rack labels; descheduler evicts for rebalancing;
        cluster autoscaler interacts with cloud APIs; vertical pod autoscaler adjusts resource requests over time.
        """,
    ]
    # for idx, ref_text in enumerate(reference_corpus, start=1):
    #     rag.add_reference_material(ref_text, {"source": f"RefDoc{idx}", "teacher_id": "T1"})

    # Direct hybrid search test (lexical vs dense):
    query = "virtualization hypervisors resource pooling"
    results = rag.hybrid_search("reference", query, k_dense=8, k_sparse=24, k_final=10)
    print("Hybrid results (reference):\n")
    for i, d in enumerate(results, 1):
        print(i, d.page_content[:120].replace('\\n',' ') + '...')
    import time
    topics_model = rag.extract_topics("virtualization")  # This returns a Topics model, not a list
    if topics_model and topics_model.topics:
        first_topic = topics_model.topics[0]
        print("First topic:", first_topic.title, first_topic.subtopics)
        time.sleep(2)
        material = generate_materials(query, teacher_id="T1")
        # generate_materials(material)
        # print("\nGenerated study material (first 500 chars):\n", material)


if __name__ == "__main__":
    main()