import React from 'react';
import { TrendingUp, ArrowRight, ShieldCheck, Scale } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';

export const F1GapRecoverySection: React.FC = () => {
  const { f1_gap_recovery } = CANONICAL_RESEARCH_DATA;

  return (
    <section className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-cyan mb-3">
            15 / F1 Gap Recovery Analysis
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Case B Classification: <br />
            <span className="italic text-foreground-muted">Approaching Quality While Preserving Safety</span>
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            By analyzing the full operating frontier, we investigate whether ClearRAG can recover its raw EM/F1 disadvantage against Standard RAG without sacrificing factual safety.
          </p>
        </div>

        {/* 3 Step Visual F1 Recovery Bar */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-border mb-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
            {/* Box 1: Default ClearRAG */}
            <div className="p-5 rounded-xl bg-surface-200 border border-border">
              <span className="text-xs font-mono uppercase text-foreground-muted block mb-1">
                Default ClearRAG (OP-04)
              </span>
              <div className="text-3xl font-mono text-foreground mb-2">
                0.1685 F1
              </div>
              <div className="text-xs text-accent-coral font-mono mb-2">
                Initial Gap: Δ = 0.0893
              </div>
              <p className="text-xs text-foreground-muted font-sans">
                Coverage: 27.60% | Unsupported Risk: 3.50%
              </p>
            </div>

            {/* Box 2: Relaxed Max Quality Point */}
            <div className="p-5 rounded-xl bg-surface-200 border border-accent-cyan/40 bg-accent-cyan/5">
              <span className="text-xs font-mono uppercase text-accent-cyan block mb-1">
                Relaxed Operating Point (OP-10)
              </span>
              <div className="text-3xl font-mono text-accent-cyan mb-2">
                0.2014 F1
              </div>
              <div className="text-xs text-accent-teal font-mono mb-2">
                36.84% Gap Recovered
              </div>
              <p className="text-xs text-foreground-muted font-sans">
                Coverage: 63.80% | Unsupported Risk: 12.00%
              </p>
            </div>

            {/* Box 3: Standard RAG Control */}
            <div className="p-5 rounded-xl bg-surface-200 border border-border">
              <span className="text-xs font-mono uppercase text-foreground-muted block mb-1">
                Standard RAG (Control)
              </span>
              <div className="text-3xl font-mono text-foreground mb-2">
                0.2578 F1
              </div>
              <div className="text-xs text-foreground-subtle font-mono mb-2">
                Upper Bound Control
              </div>
              <p className="text-xs text-foreground-muted font-sans">
                Coverage: 100.00% | Unsupported Risk: 37.08%
              </p>
            </div>
          </div>
        </div>

        {/* Scientific Conclusion Box */}
        <div className="p-7 rounded-2xl bg-surface-100 border border-border">
          <div className="text-xs font-mono uppercase text-accent-cyan tracking-wider mb-2">
            Formal Research Classification: Case B
          </div>
          <h3 className="text-lg font-medium text-foreground mb-3">
            ClearRAG approaches Standard RAG answer quality while retaining substantial safety superiority.
          </h3>
          <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-4">
            At OP-10 (63.8% coverage), ClearRAG achieves a <strong>67.6% relative reduction in unsupported claims</strong> (12.00% vs 37.08%) and maintains 85.00% verifiable attribution, while closing over a third of the token F1 gap against unconstrained Standard RAG.
          </p>
          <p className="text-xs text-foreground-subtle font-mono border-t border-border/60 pt-3">
            Note: Matching Standard RAG’s 0.2578 F1 would require disabling verification gating completely, re-introducing the 37.08% hallucination rate.
          </p>
        </div>
      </div>
    </section>
  );
};
