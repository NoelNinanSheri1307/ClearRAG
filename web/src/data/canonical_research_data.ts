/**
 * ClearRAG Single Source of Truth Canonical Research Data.
 * 
 * Sourced strictly from:
 * - results/final_canonical_evaluation.json
 * - results/coverage_risk_quality.json
 * - results/generation_experiments.json
 * - docs/final_reproducibility_audit.md
 */

export interface SystemMetrics {
  system_name: string;
  short_name: string;
  description: string;
  retrieval_strategy: string;
  verification_layer: string;
  generation_policy: string;
  gold_retrieval_success_rate: number;
  unrecoverable_retrieval_failures: number;
  verification_accuracy: number;
  answer_coverage_rate: number;
  total_answers_generated: number;
  total_abstentions: number;
  generated_exact_match: number;
  generated_token_f1: number;
  all_instances_exact_match: number;
  all_instances_token_f1: number;
  supported_claim_rate: number;
  unsupported_claim_rate: number;
  attribution_coverage: number;
  attribution_precision: number;
  safe_decision_rate: number;
  safe_abstention_rate: number;
  unsafe_answer_rate: number;
  oracle_safe_gap: number;
  llm_calls_count: number;
  llm_calls_avoided: number;
  compute_saved_percentage: number;
  mean_total_latency_ms: number;
}

export interface OperatingPoint {
  name: string;
  sim_threshold: number;
  overlap_ratio: number;
  answer_coverage_rate: number;
  answered_count: number;
  abstained_count: number;
  answered_exact_match: number;
  answered_token_f1: number;
  all_instances_exact_match: number;
  all_instances_token_f1: number;
  unsupported_claim_rate: number;
  supported_claim_rate: number;
  attribution_coverage: number;
  faithfulness_score: number;
  correct_abstention_rate: number;
  unsafe_answer_rate: number;
  compute_saved_percentage: number;
  mean_pipeline_latency_ms: number;
  composite_utility: number;
  category?: 'ultra_safe' | 'strict' | 'default' | 'balanced' | 'quality' | 'broad';
}

export interface GenerationExperiment {
  id: string;
  name: string;
  strategy: string;
  exact_match: number;
  token_f1: number;
  supported_claim_rate: number;
  unsupported_claim_rate: number;
  attribution_coverage: number;
  faithfulness_score: number;
  caveat_compliance?: number;
}

export interface MetricDefinitionItem {
  id: string;
  name: string;
  category: 'Retrieval' | 'Verification' | 'Decision & Safety' | 'Generation Quality' | 'Attribution' | 'Efficiency' | 'Statistics';
  formula?: string;
  plain_english: string;
  higher_is_better: boolean;
  standard_rag_value: string;
  clearrag_default_value: string;
  interpretation: string;
}

export const CANONICAL_RESEARCH_DATA = {
  meta: {
    project_name: 'ClearRAG',
    full_title: 'Evidence-Aware Selective Retrieval-Augmented Generation',
    version: '1.0.0-frozen',
    dataset_name: '1,250-Query Controlled Multi-Hop Evaluation Benchmark (HotpotQA-derived)',
    total_queries: 1250,
    hardware: 'NVIDIA GeForce RTX 2050 GPU (4GB VRAM)',
    generator_model: 'Qwen/Qwen2.5-1.5B-Instruct',
    embedding_model: 'BAAI/bge-small-en-v1.5',
    verified_test_count: 103,
  },

  benchmark: {
    total: 1250,
    domains: [
      {
        name: 'Supported Domain',
        total: 500,
        description: 'Evidence contains sufficient factual grounding to answer completely or partially.',
        conditions: [
          { name: 'full_evidence', count: 250, description: 'All required entity and bridge relations present in retrieved passages.' },
          { name: 'partial_evidence', count: 250, description: 'Only one bridge entity or attribute present; requires caveat-aware synthesis.' },
        ],
      },
      {
        name: 'Unanswerable Domain',
        total: 500,
        description: 'Retrieved evidence is strictly missing or directly contradictory.',
        conditions: [
          { name: 'unsupported', count: 250, description: 'No factual support for target entity in retrieved text; requires safe abstention.' },
          { name: 'conflict', count: 250, description: 'Contradictory dates, numbers, or attributes across retrieved sources; requires conflict preservation.' },
        ],
      },
      {
        name: 'Distractor Domain',
        total: 250,
        description: 'Evidence contains misleading or topically related distractor passages alongside or without facts.',
        conditions: [
          { name: 'distractor_heavy', count: 250, description: 'High-overlap surface text designed to mislead standard dense semantic retrievers.' },
        ],
      },
    ],
  },

  canonical_systems: {
    system_0: {
      system_name: 'System 0: Standard RAG (Frozen Control)',
      short_name: 'Standard RAG (Control)',
      description: 'Dense BGE-small (k=5), Always-Answer policy, unconstrained generation prompt without evidence verification.',
      retrieval_strategy: 'Dense BGE-small (k=5)',
      verification_layer: 'None (Always answers)',
      generation_policy: 'Always-Answer Unconstrained',
      gold_retrieval_success_rate: 69.12,
      unrecoverable_retrieval_failures: 386,
      verification_accuracy: 0.0,
      answer_coverage_rate: 100.0,
      total_answers_generated: 1250,
      total_abstentions: 0,
      generated_exact_match: 11.68,
      generated_token_f1: 0.2578,
      all_instances_exact_match: 11.68,
      all_instances_token_f1: 0.2578,
      supported_claim_rate: 62.92,
      unsupported_claim_rate: 37.08,
      attribution_coverage: 0.0,
      attribution_precision: 0.0,
      safe_decision_rate: 28.40,
      safe_abstention_rate: 0.0,
      unsafe_answer_rate: 100.0,
      oracle_safe_gap: 60.0,
      llm_calls_count: 1250,
      llm_calls_avoided: 0,
      compute_saved_percentage: 0.0,
      mean_total_latency_ms: 2490.0,
    } as SystemMetrics,

    system_1: {
      system_name: 'System 1: Baseline ClearRAG',
      short_name: 'Baseline ClearRAG',
      description: 'Dense BGE-small (k=5) with rule-based claim verification and basic abstention decision.',
      retrieval_strategy: 'Dense BGE-small (k=5)',
      verification_layer: 'Rule-Based Claim Extractor',
      generation_policy: 'Evidence-Gated Basic',
      gold_retrieval_success_rate: 69.12,
      unrecoverable_retrieval_failures: 386,
      verification_accuracy: 26.20,
      answer_coverage_rate: 70.96,
      total_answers_generated: 887,
      total_abstentions: 363,
      generated_exact_match: 5.98,
      generated_token_f1: 0.1670,
      all_instances_exact_match: 4.24,
      all_instances_token_f1: 0.1188,
      supported_claim_rate: 81.20,
      unsupported_claim_rate: 18.80,
      attribution_coverage: 58.40,
      attribution_precision: 86.40,
      safe_decision_rate: 42.40,
      safe_abstention_rate: 29.04,
      unsafe_answer_rate: 70.96,
      oracle_safe_gap: 21.4,
      llm_calls_count: 887,
      llm_calls_avoided: 363,
      compute_saved_percentage: 29.04,
      mean_total_latency_ms: 2520.54,
    } as SystemMetrics,

    system_2: {
      system_name: 'System 2: Retrieval-Improved ClearRAG',
      short_name: 'Retrieval-Improved',
      description: 'Hybrid Dense+BM25 with RRF and CrossScorer reranking (k=10), improving evidence recall from 69.1% to 87.8%.',
      retrieval_strategy: 'Hybrid Dense+BM25 RRF + CrossScorer (k=10)',
      verification_layer: 'Rule-Based Claim Extractor',
      generation_policy: 'Evidence-Gated Basic',
      gold_retrieval_success_rate: 87.84,
      unrecoverable_retrieval_failures: 152,
      verification_accuracy: 26.20,
      answer_coverage_rate: 72.56,
      total_answers_generated: 907,
      total_abstentions: 343,
      generated_exact_match: 6.17,
      generated_token_f1: 0.1706,
      all_instances_exact_match: 4.48,
      all_instances_token_f1: 0.1238,
      supported_claim_rate: 82.50,
      unsupported_claim_rate: 17.50,
      attribution_coverage: 61.20,
      attribution_precision: 88.20,
      safe_decision_rate: 43.80,
      safe_abstention_rate: 27.44,
      unsafe_answer_rate: 72.56,
      oracle_safe_gap: 19.8,
      llm_calls_count: 907,
      llm_calls_avoided: 343,
      compute_saved_percentage: 27.44,
      mean_total_latency_ms: 2518.20,
    } as SystemMetrics,

    system_3: {
      system_name: 'System 3: Verification-Improved ClearRAG',
      short_name: 'Verification-Improved',
      description: 'Semantic embedding similarity, non-stopword overlap, multi-hop relation verification, and contradiction detection.',
      retrieval_strategy: 'Hybrid Dense+BM25 RRF + CrossScorer (k=10)',
      verification_layer: 'Semantic & Contradiction Verifier',
      generation_policy: 'Calibrated Evidence-Gated',
      gold_retrieval_success_rate: 87.84,
      unrecoverable_retrieval_failures: 152,
      verification_accuracy: 44.80,
      answer_coverage_rate: 27.60,
      total_answers_generated: 345,
      total_abstentions: 905,
      generated_exact_match: 6.67,
      generated_token_f1: 0.1685,
      all_instances_exact_match: 1.84,
      all_instances_token_f1: 0.0465,
      supported_claim_rate: 96.80,
      unsupported_claim_rate: 3.20,
      attribution_coverage: 65.20,
      attribution_precision: 89.10,
      safe_decision_rate: 61.80,
      safe_abstention_rate: 71.60,
      unsafe_answer_rate: 28.40,
      oracle_safe_gap: 6.2,
      llm_calls_count: 345,
      llm_calls_avoided: 905,
      compute_saved_percentage: 72.40,
      mean_total_latency_ms: 730.59,
    } as SystemMetrics,

    system_4: {
      system_name: 'System 4: Final Grounded ClearRAG (Default Calibrated OP-04)',
      short_name: 'Final ClearRAG (Default)',
      description: 'Full pipeline integrating Hybrid RRF retrieval, Semantic Verifier, Gated Decision, and Grounded Citation Synthesis.',
      retrieval_strategy: 'Hybrid Dense+BM25 RRF + CrossScorer (k=10)',
      verification_layer: 'Improved Semantic & Contradiction Verifier',
      generation_policy: 'Grounded Citation Synthesis + Caveat Synthesis',
      gold_retrieval_success_rate: 87.84,
      unrecoverable_retrieval_failures: 152,
      verification_accuracy: 44.80,
      answer_coverage_rate: 27.60,
      total_answers_generated: 345,
      total_abstentions: 905,
      generated_exact_match: 6.67,
      generated_token_f1: 0.1685,
      all_instances_exact_match: 1.84,
      all_instances_token_f1: 0.0465,
      supported_claim_rate: 96.80,
      unsupported_claim_rate: 3.20,
      attribution_coverage: 94.50,
      attribution_precision: 95.20,
      safe_decision_rate: 61.80,
      safe_abstention_rate: 71.60,
      unsafe_answer_rate: 28.40,
      oracle_safe_gap: 6.2,
      llm_calls_count: 345,
      llm_calls_avoided: 905,
      compute_saved_percentage: 72.40,
      mean_total_latency_ms: 730.59,
    } as SystemMetrics,
  },

  generation_ablation_experiments: [
    {
      id: 'G-A',
      name: 'Generation Control',
      strategy: 'Standard generation prompt without grounding constraints on verified evidence subset.',
      exact_match: 6.84,
      token_f1: 0.1892,
      supported_claim_rate: 83.42,
      unsupported_claim_rate: 16.58,
      attribution_coverage: 65.20,
      faithfulness_score: 84.12,
    },
    {
      id: 'G-B',
      name: 'Evidence-Only Grounding',
      strategy: 'Strictly restricts LLM context to verified supporting evidence chunks.',
      exact_match: 7.12,
      token_f1: 0.1945,
      supported_claim_rate: 91.80,
      unsupported_claim_rate: 8.20,
      attribution_coverage: 78.40,
      faithfulness_score: 90.25,
    },
    {
      id: 'G-C',
      name: 'Claim-Level Attribution',
      strategy: 'Requires bracketed citation anchors [1], [2] matching evidence chunk IDs.',
      exact_match: 7.39,
      token_f1: 0.1988,
      supported_claim_rate: 95.10,
      unsupported_claim_rate: 4.90,
      attribution_coverage: 92.60,
      faithfulness_score: 94.70,
    },
    {
      id: 'G-D',
      name: 'Caveat-Aware Synthesis',
      strategy: 'Includes structured hedging prefixes when partial evidence is detected.',
      exact_match: 7.52,
      token_f1: 0.2014,
      supported_claim_rate: 96.80,
      unsupported_claim_rate: 3.20,
      attribution_coverage: 94.50,
      faithfulness_score: 96.15,
      caveat_compliance: 98.40,
    },
    {
      id: 'G-E',
      name: 'Conflict-Aware Synthesis',
      strategy: 'Preserves divergent factual perspectives rather than picking arbitrary single sources.',
      exact_match: 7.52,
      token_f1: 0.2014,
      supported_claim_rate: 96.80,
      unsupported_claim_rate: 3.20,
      attribution_coverage: 94.50,
      faithfulness_score: 96.15,
    },
    {
      id: 'G-F',
      name: 'Final Grounded Synthesis',
      strategy: 'Full combined grounded generation pipeline with multi-source attribution.',
      exact_match: 7.52,
      token_f1: 0.2014,
      supported_claim_rate: 96.80,
      unsupported_claim_rate: 3.20,
      attribution_coverage: 94.50,
      faithfulness_score: 96.15,
    },
  ] as GenerationExperiment[],

  operating_points_sweep: [
    {
      name: 'Setting 1: Ultra-Safe (θ=0.90)',
      sim_threshold: 0.90,
      overlap_ratio: 0.50,
      answer_coverage_rate: 14.80,
      answered_count: 185,
      abstained_count: 1065,
      answered_exact_match: 7.10,
      answered_token_f1: 0.1920,
      all_instances_exact_match: 0.0284,
      all_instances_token_f1: 0.0284,
      unsupported_claim_rate: 0.80,
      supported_claim_rate: 99.20,
      attribution_coverage: 96.20,
      faithfulness_score: 99.12,
      correct_abstention_rate: 92.00,
      unsafe_answer_rate: 8.00,
      compute_saved_percentage: 85.20,
      mean_pipeline_latency_ms: 433.27,
      composite_utility: 0.3046,
      category: 'ultra_safe',
    },
    {
      name: 'Setting 2: Strict (θ=0.85)',
      sim_threshold: 0.85,
      overlap_ratio: 0.45,
      answer_coverage_rate: 21.40,
      answered_count: 268,
      abstained_count: 982,
      answered_exact_match: 7.10,
      answered_token_f1: 0.1920,
      all_instances_exact_match: 0.0411,
      all_instances_token_f1: 0.0411,
      unsupported_claim_rate: 1.55,
      supported_claim_rate: 98.45,
      attribution_coverage: 96.20,
      faithfulness_score: 98.30,
      correct_abstention_rate: 88.00,
      unsafe_answer_rate: 12.00,
      compute_saved_percentage: 78.60,
      mean_pipeline_latency_ms: 586.58,
      composite_utility: 0.3086,
      category: 'strict',
    },
    {
      name: 'Setting 4: Default Calibrated (θ=0.75)',
      sim_threshold: 0.75,
      overlap_ratio: 0.35,
      answer_coverage_rate: 27.60,
      answered_count: 345,
      abstained_count: 905,
      answered_exact_match: 6.67,
      answered_token_f1: 0.1685,
      all_instances_exact_match: 1.84,
      all_instances_token_f1: 0.0465,
      unsupported_claim_rate: 3.50,
      supported_claim_rate: 96.50,
      attribution_coverage: 94.50,
      faithfulness_score: 96.15,
      correct_abstention_rate: 75.00,
      unsafe_answer_rate: 24.20,
      compute_saved_percentage: 72.40,
      mean_pipeline_latency_ms: 730.59,
      composite_utility: 0.2557,
      category: 'default',
    },
    {
      name: 'Setting 6: Moderate (θ=0.65)',
      sim_threshold: 0.65,
      overlap_ratio: 0.25,
      answer_coverage_rate: 35.80,
      answered_count: 448,
      abstained_count: 802,
      answered_exact_match: 7.12,
      answered_token_f1: 0.1935,
      all_instances_exact_match: 2.55,
      all_instances_token_f1: 0.0693,
      unsupported_claim_rate: 5.50,
      supported_claim_rate: 94.50,
      attribution_coverage: 91.50,
      faithfulness_score: 93.95,
      correct_abstention_rate: 68.00,
      unsafe_answer_rate: 32.00,
      compute_saved_percentage: 64.20,
      mean_pipeline_latency_ms: 921.06,
      composite_utility: 0.2689,
      category: 'balanced',
    },
    {
      name: 'Setting 9: Balanced Pareto (θ=0.50)',
      sim_threshold: 0.50,
      overlap_ratio: 0.20,
      answer_coverage_rate: 58.40,
      answered_count: 730,
      abstained_count: 520,
      answered_exact_match: 7.35,
      answered_token_f1: 0.1980,
      all_instances_exact_match: 4.29,
      all_instances_token_f1: 0.1156,
      unsupported_claim_rate: 9.25,
      supported_claim_rate: 90.75,
      attribution_coverage: 90.00,
      faithfulness_score: 89.82,
      correct_abstention_rate: 63.00,
      unsafe_answer_rate: 42.50,
      compute_saved_percentage: 41.60,
      mean_pipeline_latency_ms: 1446.02,
      composite_utility: 0.2894,
      category: 'balanced',
    },
    {
      name: 'Setting 10: Max Quality (θ=0.45)',
      sim_threshold: 0.45,
      overlap_ratio: 0.15,
      answer_coverage_rate: 63.80,
      answered_count: 798,
      abstained_count: 452,
      answered_exact_match: 7.52,
      answered_token_f1: 0.2014,
      all_instances_exact_match: 4.80,
      all_instances_token_f1: 0.1285,
      unsupported_claim_rate: 12.00,
      supported_claim_rate: 88.00,
      attribution_coverage: 85.00,
      faithfulness_score: 86.80,
      correct_abstention_rate: 52.00,
      unsafe_answer_rate: 48.00,
      compute_saved_percentage: 36.20,
      mean_pipeline_latency_ms: 1571.45,
      composite_utility: 0.2738,
      category: 'quality',
    },
    {
      name: 'Setting 12: Max Coverage (θ=0.30)',
      sim_threshold: 0.30,
      overlap_ratio: 0.10,
      answer_coverage_rate: 84.20,
      answered_count: 1052,
      abstained_count: 198,
      answered_exact_match: 7.40,
      answered_token_f1: 0.1984,
      all_instances_exact_match: 6.23,
      all_instances_token_f1: 0.1671,
      unsupported_claim_rate: 17.25,
      supported_claim_rate: 82.75,
      attribution_coverage: 82.00,
      faithfulness_score: 81.02,
      correct_abstention_rate: 40.00,
      unsafe_answer_rate: 60.00,
      compute_saved_percentage: 15.80,
      mean_pipeline_latency_ms: 2045.30,
      composite_utility: 0.2667,
      category: 'broad',
    },
  ] as OperatingPoint[],

  statistical_validation: {
    mcnemar: {
      test_name: "McNemar's Paired Test with Continuity Correction",
      metric: 'Response Safety on Paired Benchmark Queries',
      b_count: 154,
      c_count: 298,
      chi2: 44.82,
      p_value: '1.01 × 10⁻¹⁴',
      p_value_raw: 1.01e-14,
      is_significant: true,
      odds_ratio: 1.93,
      interpretation: 'ClearRAG is 1.93x more likely to deliver a safe decision than Standard RAG on identical benchmark queries (p < 0.001).',
    },
    wilcoxon: {
      test_name: 'Wilcoxon Signed-Rank Paired Test',
      metric: 'All-Instances Token F1 Paired Distribution',
      w_stat: 12450.0,
      p_value: '5.30 × 10⁻⁹³',
      p_value_raw: 5.30e-93,
      is_significant: true,
      cohens_d: -0.606,
      interpretation: 'Statistically significant difference in paired Token F1 resulting from ClearRAG scoring 0.0 on abstained queries (effect size d = -0.606).',
    },
    bootstrap_95_ci: {
      resamples: 1000,
      metrics: [
        { name: 'Supported Claim Rate', clearrag_ci: '[95.80%, 97.60%]', standard_rag_ci: '[60.10%, 65.40%]' },
        { name: 'Attribution Coverage', clearrag_ci: '[93.20%, 95.80%]', standard_rag_ci: '[0.00%, 0.00%]' },
        { name: 'Safe Abstention Rate', clearrag_ci: '[67.40%, 75.60%]', standard_rag_ci: '[0.00%, 0.00%]' },
        { name: 'Mean Pipeline Latency (ms)', clearrag_ci: '[695.2 ms, 766.4 ms]', standard_rag_ci: '[2460.0 ms, 2520.0 ms]' },
      ],
    },
  },

  f1_gap_recovery: {
    standard_rag_f1: 0.2578,
    clearrag_default_f1: 0.1685,
    initial_gap: 0.0893,
    clearrag_max_quality_f1: 0.2014,
    remaining_gap_at_op10: 0.0564,
    f1_gap_recovered_pct: 36.84,
    unsupported_reduction_at_op10_pct: 67.64,
    classification: 'Case B: ClearRAG approaches Standard RAG quality while preserving major factual safety advantages.',
  },

  metric_dictionary: [
    {
      id: 'answer_coverage',
      name: 'Answer Coverage Rate',
      category: 'Decision & Safety',
      formula: 'Coverage = (Answered Queries / Total Benchmark Queries) × 100%',
      plain_english: 'The percentage of benchmark questions for which the system attempts to generate an answer rather than abstaining.',
      higher_is_better: false,
      standard_rag_value: '100.00%',
      clearrag_default_value: '27.60%',
      interpretation: 'Standard RAG always answers (100%). ClearRAG gates generation on verified sufficient evidence, answering selectively.',
    },
    {
      id: 'exact_match_answered',
      name: 'Exact Match (Answered-Instance)',
      category: 'Generation Quality',
      formula: 'EM = (Exact Prediction/Reference Matches / Total Answered Instances) × 100%',
      plain_english: 'The percentage of generated answers that match the normalized gold reference string character-for-character.',
      higher_is_better: true,
      standard_rag_value: '11.68%',
      clearrag_default_value: '6.67%',
      interpretation: 'Measures lexical precision on answered queries. Standard RAG scores higher due to unconstrained lexical matching.',
    },
    {
      id: 'token_f1_answered',
      name: 'Token F1 (Answered-Instance)',
      category: 'Generation Quality',
      formula: 'F1 = 2 × (Precision × Recall) / (Precision + Recall) across prediction and gold tokens',
      plain_english: 'Harmonic mean of token-level precision and recall between generated answers and reference answers on answered queries.',
      higher_is_better: true,
      standard_rag_value: '0.2578',
      clearrag_default_value: '0.1685',
      interpretation: 'Standard RAG generates verbose topical tokens matching gold references; ClearRAG generates concise, cited sentences.',
    },
    {
      id: 'all_instances_f1',
      name: 'Token F1 (All-Instances)',
      category: 'Generation Quality',
      formula: 'All_F1 = (Sum of F1 on answered queries + 0.0 for abstentions) / 1,250',
      plain_english: 'Average Token F1 across all 1,250 queries, where every abstained query receives a penalty score of 0.0.',
      higher_is_better: true,
      standard_rag_value: '0.2578',
      clearrag_default_value: '0.0465',
      interpretation: 'Reflects the mathematical tradeoff: All_F1 = Answered_F1 × Coverage (0.1685 × 0.2760 ≈ 0.0465).',
    },
    {
      id: 'unsupported_claim_rate',
      name: 'Unsupported Claim Rate',
      category: 'Decision & Safety',
      formula: 'Unsupported_Rate = (Hallucinated Claims / Total Generated Claims) × 100%',
      plain_english: 'The proportion of factual statements generated by the model that have no backing in the retrieved context.',
      higher_is_better: false,
      standard_rag_value: '37.08%',
      clearrag_default_value: '3.20%',
      interpretation: 'ClearRAG eliminates 91.4% of unsupported assertions compared to Standard RAG.',
    },
    {
      id: 'supported_claim_rate',
      name: 'Supported Claim Rate',
      category: 'Decision & Safety',
      formula: 'Supported_Rate = (Verified Grounded Claims / Total Generated Claims) × 100%',
      plain_english: 'The percentage of generated statements that are directly verified against retrieved evidence chunks.',
      higher_is_better: true,
      standard_rag_value: '62.92%',
      clearrag_default_value: '96.80%',
      interpretation: 'Over 96% of claims generated by ClearRAG are fully supported by evidence.',
    },
    {
      id: 'attribution_coverage',
      name: 'Attribution Coverage',
      category: 'Attribution',
      formula: 'Attribution_Coverage = (Claims with Valid Citations / Total Generated Claims) × 100%',
      plain_english: 'The percentage of generated statements that feature explicit, verifiable citation markers [1], [2] pointing to evidence passages.',
      higher_is_better: true,
      standard_rag_value: '0.00%',
      clearrag_default_value: '94.50%',
      interpretation: 'Standard RAG provides no explicit sentence-level provenance; ClearRAG attributes 94.5% of claims.',
    },
    {
      id: 'correct_safe_abstention',
      name: 'Correct Safe Abstention Rate',
      category: 'Decision & Safety',
      formula: 'Safe_Abstention = (Abstentions on Unanswerable Queries / 500 Unanswerable Queries) × 100%',
      plain_english: 'How often the system correctly refuses to answer when evidence is missing (unsupported) or contradictory (conflict).',
      higher_is_better: true,
      standard_rag_value: '0.00%',
      clearrag_default_value: '71.60%',
      interpretation: 'Standard RAG guesses or hallucinates on 100% of unanswerables; ClearRAG safely abstains on 358 of 500.',
    },
    {
      id: 'unsafe_answer_rate',
      name: 'Unsafe Answer Rate',
      category: 'Decision & Safety',
      formula: 'Unsafe_Answer_Rate = (Answers on Unanswerable Queries / 500 Unanswerable Queries) × 100%',
      plain_english: 'How often the system produces an answer when it should have abstained due to missing or contradictory evidence.',
      higher_is_better: false,
      standard_rag_value: '100.00%',
      clearrag_default_value: '28.40%',
      interpretation: 'ClearRAG cuts unsafe failure modes from 100% (500/500) down to 28.4% (142/500).',
    },
    {
      id: 'gold_retrieval_success',
      name: 'Gold Retrieval Success Rate',
      category: 'Retrieval',
      formula: 'Retrieval_Success = (Queries with All Gold Passages Retrieved / Total Queries) × 100%',
      plain_english: 'The percentage of queries where the retrieval stage successfully placed all necessary gold bridge evidence into top-k context.',
      higher_is_better: true,
      standard_rag_value: '69.12%',
      clearrag_default_value: '87.84%',
      interpretation: 'Hybrid Dense+BM25 with CrossScorer reranking (k=10) reduced unrecoverable retrieval failures from 386 to 152 (-60.6%).',
    },
    {
      id: 'verification_accuracy',
      name: 'Verification Accuracy',
      category: 'Verification',
      formula: 'Verification_Acc = (Correct Claim & Sufficiency Classifications / Total Evaluated Queries) × 100%',
      plain_english: 'The accuracy of the verification engine in classifying whether retrieved evidence is fully supported, partial, unsupported, or conflicting.',
      higher_is_better: true,
      standard_rag_value: 'N/A (0.0%)',
      clearrag_default_value: '44.80%',
      interpretation: 'Improved verifier combines semantic embeddings, non-stopword overlap, multi-hop relation checking, and contradiction detection.',
    },
    {
      id: 'compute_saved',
      name: 'LLM Generation Compute Saved',
      category: 'Efficiency',
      formula: 'Compute_Saved = (Avoided LLM Calls / Total Benchmark Queries) × 100%',
      plain_english: 'The percentage of expensive GPU LLM generation calls avoided through deterministic early abstention at the verifier layer.',
      higher_is_better: true,
      standard_rag_value: '0.00%',
      clearrag_default_value: '72.40%',
      interpretation: 'Avoids 905 GPU LLM generation calls, reducing compute costs and average pipeline latency by 70.7%.',
    },
    {
      id: 'mean_latency',
      name: 'Mean Total Pipeline Latency',
      category: 'Efficiency',
      formula: 'Mean_Latency = Total End-to-End Elapsed Time / Total Benchmark Queries',
      plain_english: 'The average time taken from question input to response output (including retrieval, verification, and generation).',
      higher_is_better: false,
      standard_rag_value: '2490.0 ms',
      clearrag_default_value: '730.6 ms',
      interpretation: 'Standard RAG takes ~2.49s on every query. ClearRAG averages 730.6ms because 72.4% of queries exit at the verifier in ~89.5ms.',
    },
  ] as MetricDefinitionItem[],

  publication_plots: [
    {
      filename: 'utility_comparison.png',
      title: 'Composite Utility Comparison',
      caption: 'Multi-objective utility balancing answer quality against factual hallucination penalties.',
      category: 'Comparative Summary',
    },
    {
      filename: 'unsupported_claim_rate_comparison.png',
      title: 'Unsupported Claim Rate Comparison',
      caption: '91.4% relative reduction in unsupported hallucinated claims (37.08% vs 3.20%).',
      category: 'Safety & Grounding',
    },
    {
      filename: 'attribution_coverage_comparison.png',
      title: 'Attribution Coverage Comparison',
      caption: '94.50% verifiable claim-to-evidence citation coverage in ClearRAG vs 0.0% in Standard RAG.',
      category: 'Attribution',
    },
    {
      filename: 'correct_abstention_comparison.png',
      title: 'Safe Abstention on Unanswerable Queries',
      caption: '71.60% correct safe abstention on unsupported and conflicting queries (358 / 500).',
      category: 'Safety & Grounding',
    },
    {
      filename: 'coverage_risk_curve.png',
      title: 'Coverage-Risk Operating Frontier',
      caption: 'Factual hallucination risk as a function of answer volume across verifier thresholds.',
      category: 'Operating Frontier',
    },
    {
      filename: 'coverage_vs_f1.png',
      title: 'Answer Coverage vs Token F1',
      caption: 'Comparison of Answered-Instance F1 vs All-Instances F1 across operating points.',
      category: 'Operating Frontier',
    },
    {
      filename: 'coverage_vs_unsupported_claim_rate.png',
      title: 'Coverage vs Unsupported Claim Rate',
      caption: 'Monotonic rise in factual risk as decision thresholds relax to admit more queries.',
      category: 'Operating Frontier',
    },
    {
      filename: 'coverage_vs_unsafe_answer_rate.png',
      title: 'Coverage vs Unsafe Answer Rate',
      caption: 'Failure rate on unanswerable queries across varying verification operating points.',
      category: 'Operating Frontier',
    },
    {
      filename: 'coverage_vs_attribution_coverage.png',
      title: 'Coverage vs Attribution Coverage',
      caption: 'Attribution density maintained above 82% across all operating points.',
      category: 'Attribution',
    },
    {
      filename: 'coverage_vs_compute_saved.png',
      title: 'Coverage vs GPU Compute Saved',
      caption: 'Proportion of avoided LLM generator calls across decision thresholds.',
      category: 'Efficiency',
    },
    {
      filename: 'risk_quality_frontier.png',
      title: 'Risk-Quality Pareto Frontier',
      caption: 'Answered-instance Token F1 plotted against unsupported claim rate.',
      category: 'Pareto Analysis',
    },
    {
      filename: 'combined_utility_frontier.png',
      title: 'Combined Multi-Objective Utility Frontier',
      caption: 'Identification of the optimal balanced operating point (OP-09 at 58.4% coverage).',
      category: 'Pareto Analysis',
    },
    {
      filename: 'error_transition_matrix.png',
      title: 'Standard RAG to ClearRAG Error Transition Matrix',
      caption: 'Query-level state transitions across all 1,250 benchmark queries.',
      category: 'Error Transitions',
    },
    {
      filename: 'systems_0_to_4_comparison.png',
      title: 'System 0 through System 4 Milestone Progression',
      caption: 'Retrieval, verification, and safety progression across architectural generations.',
      category: 'Milestone Evolution',
    },
  ],
};
