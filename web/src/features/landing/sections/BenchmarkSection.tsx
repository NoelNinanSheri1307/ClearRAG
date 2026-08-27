import React from 'react';
import { Database, FileText, CheckCircle, AlertTriangle, XOctagon, Layers, BookOpen } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';

export const BenchmarkSection: React.FC = () => {
  return (
    <section id="benchmark" className="py-24 px-6 border-t border-border/60 bg-surface-300/30">
      <div className="max-w-5xl mx-auto">
        {/* Section Header */}
        <div className="mb-12">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            03 / The Evaluation Dataset
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            HotpotQA Multi-Hop Benchmark
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            The evaluation is built upon <strong>HotpotQA</strong>—a premier multi-hop question answering benchmark derived from Wikipedia. Multi-hop questions require combining factual assertions across multiple disparate passages to arrive at a correct answer.
          </p>
        </div>

        {/* HotpotQA Dataset Foundation Card */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-border mb-12">
          <div className="flex items-center justify-between mb-6 pb-4 border-b border-border/60">
            <div>
              <span className="text-xs font-mono uppercase text-accent-teal tracking-wider block">
                Dataset Foundation
              </span>
              <h3 className="text-lg font-medium text-foreground">
                Why Multi-Hop HotpotQA?
              </h3>
            </div>
            <span className="text-xs font-mono text-foreground-muted px-2.5 py-1 rounded bg-surface-200 border border-border">
              Multi-Document Reasoning
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <strong className="text-xs font-mono text-foreground block mb-1">
                Multi-Hop Bridge Reasoning
              </strong>
              <p className="text-[11px] font-sans text-foreground-muted leading-relaxed">
                Questions cannot be answered from a single sentence or document; they require chaining facts across two or more supporting Wikipedia passages.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <strong className="text-xs font-mono text-foreground block mb-1">
                Natural Distractor Passages
              </strong>
              <p className="text-[11px] font-sans text-foreground-muted leading-relaxed">
                Each question comes embedded with 8-10 topically related distractor passages designed to deceive keyword matching and test verifier precision.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <strong className="text-xs font-mono text-foreground block mb-1">
                1,250 Curated Queries
              </strong>
              <p className="text-[11px] font-sans text-foreground-muted leading-relaxed">
                Structured into 5 controlled experimental conditions (250 questions each) to evaluate system behavior across every evidence scenario.
              </p>
            </div>
          </div>
        </div>

        {/* 3 Domain Breakdown Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {/* Supported Domain */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-accent-teal tracking-wider">
                  Supported Domain
                </span>
                <span className="text-sm font-mono text-foreground font-medium">
                  500 Queries
                </span>
              </div>
              <p className="text-xs text-foreground-muted mb-6 leading-relaxed">
                Evidence contains factual support to answer completely or with structured caveats.
              </p>

              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded-lg bg-surface-200 border border-border/60">
                  <div className="flex justify-between text-foreground mb-1">
                    <span>full_evidence</span>
                    <span className="text-accent-teal">250</span>
                  </div>
                  <p className="text-[11px] font-sans text-foreground-muted">All required bridge entity facts present.</p>
                </div>
                <div className="p-3 rounded-lg bg-surface-200 border border-border/60">
                  <div className="flex justify-between text-foreground mb-1">
                    <span>partial_evidence</span>
                    <span className="text-accent-teal">250</span>
                  </div>
                  <p className="text-[11px] font-sans text-foreground-muted">Only partial evidence; needs explicit caveats.</p>
                </div>
              </div>
            </div>
            <div className="mt-6 pt-3 border-t border-border/60 text-[11px] font-mono text-accent-teal">
              Goal: High answer quality & provenance
            </div>
          </div>

          {/* Unanswerable Domain */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-accent-coral tracking-wider">
                  Unanswerable Domain
                </span>
                <span className="text-sm font-mono text-foreground font-medium">
                  500 Queries
                </span>
              </div>
              <p className="text-xs text-foreground-muted mb-6 leading-relaxed">
                Evidence is completely missing or internally contradictory. Requires abstention.
              </p>

              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded-lg bg-surface-200 border border-border/60">
                  <div className="flex justify-between text-foreground mb-1">
                    <span>unsupported</span>
                    <span className="text-accent-coral">250</span>
                  </div>
                  <p className="text-[11px] font-sans text-foreground-muted">Missing key facts. Safe abstention required.</p>
                </div>
                <div className="p-3 rounded-lg bg-surface-200 border border-border/60">
                  <div className="flex justify-between text-foreground mb-1">
                    <span>conflict</span>
                    <span className="text-accent-coral">250</span>
                  </div>
                  <p className="text-[11px] font-sans text-foreground-muted">Contradictory claims. Disagreement preserved.</p>
                </div>
              </div>
            </div>
            <div className="mt-6 pt-3 border-t border-border/60 text-[11px] font-mono text-accent-coral">
              Goal: Safe abstention (Zero hallucination)
            </div>
          </div>

          {/* Distractor Domain */}
          <div className="p-6 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-accent-amber tracking-wider">
                  Distractor Domain
                </span>
                <span className="text-sm font-mono text-foreground font-medium">
                  250 Queries
                </span>
              </div>
              <p className="text-xs text-foreground-muted mb-6 leading-relaxed">
                Context is dominated by high-overlap distractor passages designed to test verifier robustness.
              </p>

              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded-lg bg-surface-200 border border-border/60">
                  <div className="flex justify-between text-foreground mb-1">
                    <span>distractor_heavy</span>
                    <span className="text-accent-amber">250</span>
                  </div>
                  <p className="text-[11px] font-sans text-foreground-muted">Topical keywords present without supporting facts.</p>
                </div>
              </div>
            </div>
            <div className="mt-6 pt-3 border-t border-border/60 text-[11px] font-mono text-accent-amber">
              Goal: Resist misleading surface text
            </div>
          </div>
        </div>

        {/* Accounting Statement */}
        <div className="p-4 rounded-xl bg-surface-100 border border-border flex items-center justify-between text-xs font-mono text-foreground-muted">
          <span>Total Curated Instances: 1,250 / 1,250</span>
          <span className="text-accent-teal">Fixed HotpotQA Benchmark</span>
        </div>
      </div>
    </section>
  );
};
