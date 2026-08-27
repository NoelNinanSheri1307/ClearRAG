import React from 'react';
import { AlertCircle, HelpCircle, Scale, CheckCircle2 } from 'lucide-react';

export const TradeoffSection: React.FC = () => {
  return (
    <section id="tradeoff" className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-coral mb-3">
            12 / The Research Tradeoff
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            The Cost of Safety
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            ClearRAG does not beat Standard RAG in raw Exact Match or Token F1. It is critical to understand why this difference exists and what the research actually establishes.
          </p>
        </div>

        {/* 2 Big Cards Explaining the Tradeoff */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          {/* Card 1: Why Standard RAG has higher raw EM/F1 */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-foreground-muted">
                  Standard RAG Behavior
                </span>
                <span className="text-xs font-mono text-foreground font-semibold">
                  11.68% EM • 0.2578 F1
                </span>
              </div>
              <h3 className="text-lg font-medium text-foreground mb-3">
                Why Standard RAG Scores Higher Raw F1
              </h3>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
                Standard RAG answers 100% of queries. On unsupported or unanswerable queries, its unconstrained generator produces verbose topical sentences. Because these sentences contain topical vocabulary matching reference answer strings, Standard RAG accumulates token overlap by guessing.
              </p>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                However, <strong>37.08% of its generated claims are completely unsupported</strong> by retrieved evidence.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-border/60 text-xs font-mono text-foreground-subtle">
              Result: High token overlap inflated by ungrounded guessing.
            </div>
          </div>

          {/* Card 2: Why ClearRAG scores lower raw EM/F1 */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-accent-teal/40 bg-accent-teal/5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-accent-teal">
                  ClearRAG Behavior
                </span>
                <span className="text-xs font-mono text-accent-teal font-semibold">
                  6.67% EM • 0.1685 F1
                </span>
              </div>
              <h3 className="text-lg font-medium text-foreground mb-3">
                The Explicit Safety Penalty
              </h3>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
                ClearRAG refuses to answer whenever evidence is unverified, answering only 27.60% of queries under strict calibrated gating. Under all-instances evaluation, every abstained question scores exactly <strong>0.0</strong> for Exact Match and Token F1.
              </p>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                Furthermore, ClearRAG strictly constrains generation to concise, cited assertions with bracketed markers <code className="text-foreground">[1]</code>, which penalizes raw token match against un-cited reference strings.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-border/60 text-xs font-mono text-accent-teal">
              Result: 91.4% fewer hallucinations at the cost of nominal coverage.
            </div>
          </div>
        </div>

        {/* Mathematical Proof Box */}
        <div className="p-7 rounded-2xl bg-surface-100 border border-border">
          <div className="text-xs font-mono uppercase tracking-wider text-foreground-muted mb-3">
            Mathematical Relationship: Answered vs All-Instances
          </div>
          <div className="font-mono text-xs text-foreground p-4 rounded-xl bg-surface-200 border border-border/60 mb-4 leading-relaxed overflow-x-auto">
            All-Instances F1 = Answered-Instance F1 × Answer Coverage Rate <br />
            0.1685 × (345 / 1,250) = 0.1685 × 0.2760 = 0.0465
          </div>
          <p className="text-xs text-foreground-muted font-sans leading-relaxed">
            This mathematical identity demonstrates why all-instances and answered-instance metrics must be reported separately. Reporting only all-instances conflates answer quality with decision coverage.
          </p>
        </div>
      </div>
    </section>
  );
};
