import React from 'react';
import { Search, CheckCircle2, GitBranch, Sparkles, BookOpen, ArrowRight } from 'lucide-react';

export const ArchitectureSection: React.FC = () => {
  const stages = [
    {
      num: '01',
      name: 'Hybrid Retrieval & Reranking',
      what: 'Retrieves top-k evidence passages combining dense neural embeddings and lexical BM25.',
      why: 'Dense retrieval misses rare entity names; BM25 alone misses synonyms. Hybrid RRF balances both.',
      implemented: 'BGE-small + BM25 + Reciprocal Rank Fusion + CrossScorer Diversity (k=10).',
      measured: 'Gold Evidence Retrieval Success Rate: 69.12% → 87.84% (+18.72 pp).',
    },
    {
      num: '02',
      name: 'Semantic & Contradiction Verification',
      what: 'Deconstructs queries into claims and verifies each against retrieved evidence chunks.',
      why: 'Prevents the generator from receiving false context or hallucinating on missing facts.',
      implemented: 'BGE-small cosine similarity, non-stopword overlap, multi-hop joint entity support, contradiction detector.',
      measured: 'Verification Accuracy: 44.80% (vs 0% in Standard RAG).',
    },
    {
      num: '03',
      name: 'Evidence-Gated Decision Engine',
      what: 'Determines whether to ANSWER, ANSWER_WITH_CAVEAT, ABSTAIN, or CONFLICT_ABSTAIN.',
      why: 'Prevents generation on unsupported queries and prevents arbitrary resolution of contradictions.',
      implemented: 'Deterministic calibrated state machine mapping sufficiency status to response actions.',
      measured: '71.60% Safe Abstention on unanswerables (vs 0.0% in Standard RAG).',
    },
    {
      num: '04',
      name: 'Grounded & Caveat-Aware Generation',
      what: 'Synthesizes verified answers strictly restricted to supporting evidence chunks.',
      why: 'Eliminates parametric hallucinations and inserts explicit caveats for partial evidence.',
      implemented: 'Structured citation prompting, caveat prefix synthesis, conflict-aware multi-perspective format.',
      measured: 'Unsupported Claim Rate reduced from 37.08% down to 3.20% (-91.4% relative).',
    },
    {
      num: '05',
      name: 'Claim-Level Attribution & Provenance',
      what: 'Maps every generated statement to verifiable bracketed citation markers [1], [2].',
      why: 'Enables users and downstream audit systems to inspect the exact evidentiary source of every claim.',
      implemented: 'AttributionEngine with sentence-level regex parsing and evidence chunk alignment.',
      measured: 'Attribution Coverage: 94.50% (vs 0.0% in Standard RAG).',
    },
  ];

  return (
    <section id="architecture" className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            04 / The Architecture
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            The 5-Stage ClearRAG Pipeline
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            Unlike linear RAG systems, ClearRAG decouples evidence verification and decision gating from generation, ensuring the LLM is invoked only when sufficient factual grounding is confirmed.
          </p>
        </div>

        {/* 5 Vertical Timeline Steps */}
        <div className="space-y-6">
          {stages.map((stage) => (
            <div
              key={stage.num}
              className="p-7 rounded-2xl bg-surface-100 border border-border hover:border-accent-teal/40 transition-all duration-300 relative group"
            >
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
                <div className="flex items-start gap-5">
                  <span className="text-xl font-mono font-bold text-accent-teal/80 shrink-0">
                    {stage.num}
                  </span>
                  <div>
                    <h3 className="text-lg font-medium text-foreground mb-2">
                      {stage.name}
                    </h3>
                    <p className="text-sm text-foreground-muted leading-relaxed font-sans mb-4">
                      {stage.what}
                    </p>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans border-t border-border/60 pt-4">
                      <div>
                        <strong className="text-foreground-muted block font-mono text-[11px] uppercase tracking-wider mb-1">
                          Why It Exists
                        </strong>
                        <span className="text-foreground-muted">{stage.why}</span>
                      </div>
                      <div>
                        <strong className="text-foreground-muted block font-mono text-[11px] uppercase tracking-wider mb-1">
                          Implementation
                        </strong>
                        <span className="text-foreground font-mono text-[11px]">{stage.implemented}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Metric Callout */}
                <div className="shrink-0 md:text-right border-t md:border-t-0 md:border-l border-border/60 pt-4 md:pt-0 md:pl-6">
                  <span className="text-[10px] font-mono uppercase tracking-widest text-accent-teal block mb-1">
                    Key Result
                  </span>
                  <span className="text-xs font-mono font-medium text-foreground block max-w-[200px]">
                    {stage.measured}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
