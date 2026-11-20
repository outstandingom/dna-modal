"""
Dynamic DNA Demo Knowledge Base & Learning Basis
Encodes domain knowledge, entity ontologies, and long-term customer memory personas
using the Recursive Compositional DNA Vector Architecture (HRR/VSA).
"""

import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger("dna_demo_dataset")

# 1. Company & Domain Ontology Triples
DOMAIN_KNOWLEDGE_TRIPLES = [
    # Company structure & Leadership
    ("company_growhaz", "founded_by", "person_om_uikey"),
    ("person_om_uikey", "is_founder_of", "company_growhaz"),
    ("person_om_uikey", "has_role", "lead_ai_architect"),
    ("company_growhaz", "headquarters", "location_india"),
    ("company_growhaz", "contact_email", "email_info_growhaz_gmail"),
    ("company_growhaz", "official_website", "url_growhaz_in"),

    # Services provided by Growhaz
    ("company_growhaz", "provides_service", "service_alphag5_audit"),
    ("company_growhaz", "provides_service", "service_alphag1_audit"),
    ("company_growhaz", "provides_service", "service_alphag3_audit"),
    ("company_growhaz", "provides_service", "service_appify_mobile"),
    ("company_growhaz", "provides_service", "service_static_website"),
    ("company_growhaz", "provides_service", "service_dynamic_webapp"),
    ("company_growhaz", "provides_service", "service_blockchain_verify"),
    ("company_growhaz", "provides_service", "service_workflow_automation"),

    # Security service capabilities & vulnerability protections
    ("service_alphag5_audit", "protects_against", "vuln_sql_injection"),
    ("service_alphag5_audit", "protects_against", "vuln_cross_site_scripting"),
    ("service_alphag5_audit", "protects_against", "vuln_idor_flaws"),
    ("service_alphag5_audit", "protects_against", "vuln_exposed_api_keys"),
    ("service_alphag5_audit", "scan_duration", "duration_5_minutes"),
    ("service_alphag5_audit", "delivery_time", "turnaround_24_hours"),
    ("service_alphag5_audit", "has_pricing", "price_inr_4999"),

    ("service_alphag1_audit", "protects_against", "vuln_sql_injection"),
    ("service_alphag1_audit", "protects_against", "vuln_cross_site_scripting"),
    ("service_alphag1_audit", "has_pricing", "price_inr_1999"),

    ("service_alphag3_audit", "audit_type", "type_manual_pentest"),
    ("service_alphag3_audit", "protects_against", "vuln_zero_day"),
    ("service_alphag3_audit", "has_pricing", "price_inr_9999"),

    # Mobile Appify Capabilities
    ("service_appify_mobile", "targets_platform", "platform_android"),
    ("service_appify_mobile", "targets_platform", "platform_ios"),
    ("service_appify_mobile", "includes_feature", "feature_push_notifications"),
    ("service_appify_mobile", "includes_feature", "feature_offline_caching"),
    ("service_appify_mobile", "includes_feature", "feature_biometric_login"),
    ("service_appify_mobile", "has_starter_pricing", "price_inr_15000"),
    ("service_appify_mobile", "has_dual_platform_pricing", "price_inr_25000"),

    # Web Development Capabilities
    ("service_static_website", "seo_guarantee", "score_100_lighthouse"),
    ("service_static_website", "has_pricing", "price_inr_20000"),
    ("service_dynamic_webapp", "includes_feature", "feature_custom_crm_dashboard"),
    ("service_dynamic_webapp", "includes_feature", "feature_payment_gateway"),
    ("service_dynamic_webapp", "has_starting_pricing", "price_inr_60000"),
]

# 2. Pre-Learned Customer Personas for Long-Term Memory Demonstration
DEMO_CUSTOMER_PERSONAS = [
    {
        "customer_id": "919876543210",
        "facts": {
            "has_name": "Om Uikey",
            "has_role": "Founder & Technical Lead",
            "has_company": "Growhaz",
            "has_website": "https://www.growhaz.in",
            "prefers": "service_alphag5_audit"
        },
        "history_events": [
            {"action": "purchased", "target": "service_alphag5_audit", "days_ago": 28},
            {"action": "purchased", "target": "service_alphag5_audit", "days_ago": 21},
            {"action": "purchased", "target": "service_alphag5_audit", "days_ago": 14},
            {"action": "purchased", "target": "service_alphag5_audit", "days_ago": 7},
            {"action": "purchased", "target": "service_alphag5_audit", "days_ago": 1},
        ]
    },
    {
        "customer_id": "919822334455",
        "facts": {
            "has_name": "Priya Sharma",
            "has_role": "Store Owner",
            "has_company": "GlamourStore",
            "has_website": "https://glamourstore.in",
            "prefers": "service_appify_mobile"
        },
        "history_events": [
            {"action": "inquired", "target": "service_appify_mobile", "days_ago": 10},
            {"action": "inquired", "target": "service_appify_mobile", "days_ago": 3},
        ]
    },
    {
        "customer_id": "919833445566",
        "facts": {
            "has_name": "Rahul Mehta",
            "has_role": "Chief Technology Officer",
            "has_company": "FintechCorp",
            "has_website": "https://fintechcorp.io",
            "prefers": "service_alphag3_audit"
        },
        "history_events": [
            {"action": "inquired", "target": "service_alphag3_audit", "days_ago": 15},
            {"action": "purchased", "target": "service_alphag3_audit", "days_ago": 5},
        ]
    }
]

def seed_demo_knowledge_base(system) -> Dict[str, Any]:
    """
    Ingests demo domain knowledge and pre-learned customer memories
    using the DNA vector encoding algorithm (HRR binding and assembly bundling).
    """
    if not system:
        return {"status": "error", "message": "DNA system is not available"}

    logger.info("Ingesting Demo Domain Knowledge Triples into DNA Vector Memory...")

    # 1. Encode Domain Knowledge Triples
    encoded_relations_count = 0
    for subj, rel_type, obj in DOMAIN_KNOWLEDGE_TRIPLES:
        # Register atomic concept nodes
        sub_atom = system.register_concept(value=subj.replace("_", " "), concept_type="domain_concept", custom_id=subj)
        obj_atom = system.register_concept(value=obj.replace("_", " "), concept_type="domain_concept", custom_id=obj)

        # Create holographic relation bond R_ABC
        system.relation_registry.create_relation(
            subject_id=sub_atom.id,
            relation_type=rel_type,
            object_id=obj_atom.id,
            confidence=1.0,
            strength=1.0
        )
        # Create reciprocal inverse relation
        system.relation_registry.create_relation(
            subject_id=obj_atom.id,
            relation_type=f"inverse_{rel_type}",
            object_id=sub_atom.id,
            confidence=1.0,
            strength=0.9
        )
        encoded_relations_count += 2

    # 2. Ingest Pre-Learned Customer Memory Personas
    now = datetime.datetime.now(datetime.timezone.utc)
    seeded_customers = 0

    for persona in DEMO_CUSTOMER_PERSONAS:
        cust_id = persona["customer_id"]
        # Register customer entity
        cust_atom = system.register_concept(
            value=cust_id,
            concept_type="customer",
            source="demo_seed",
            custom_id=cust_id
        )

        # Encode customer facts (Name, Role, Company, Website, etc.)
        for fact_type, fact_val in persona["facts"].items():
            val_atom = system.register_concept(
                value=str(fact_val),
                concept_type="customer_attribute",
                source="demo_seed"
            )
            # Bind relationship
            system.relation_registry.create_relation(
                subject_id=cust_id,
                relation_type=fact_type,
                object_id=val_atom.id,
                confidence=1.0
            )
            system.relation_registry.create_relation(
                subject_id=val_atom.id,
                relation_type=f"is_{fact_type}_of",
                object_id=cust_id,
                confidence=1.0
            )

        # Ingest episodic event history stream
        for evt in persona["history_events"]:
            delta = datetime.timedelta(days=evt["days_ago"])
            evt_ts = (now - delta).isoformat()
            system.record_interaction(
                customer_id=cust_id,
                event_type=evt["action"],
                target_id=evt["target"],
                attributes={"demo_seed": True},
                timestamp=evt_ts
            )

        seeded_customers += 1

    # Synchronize ANN retrieval index
    system.sync_index()

    logger.info(
        f"Seeded DNA Knowledge Base: {system.dna_registry.count()} atoms, "
        f"{len(system.relation_registry._relations)} relations, "
        f"{system.assembly_engine.count()} assemblies."
    )

    return {
        "status": "success",
        "atoms_count": system.dna_registry.count(),
        "relations_count": len(system.relation_registry._relations),
        "assemblies_count": system.assembly_engine.count(),
        "customers_seeded": seeded_customers
    }
