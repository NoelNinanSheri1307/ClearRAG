import React from 'react';
import { CheckCircle2, ShieldCheck, Zap, Sparkles, TrendingUp } from 'lucide-react';

export const SynthesisSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            18 / Research Synthesis
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Key Empirical Achievements
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            A concise summary of verified system capabilities demonstrated across the 1,250-query evaluation benchmark.
          </p>
        </div>

        {/* Verified Research Achievements Card */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-accent-teal/40 bg-accent-teal/5">
          <div className="flex items-center gap-2.5 mb-8 text-accent-teal">
            <CheckCircle2 className="w-5 h-5" />
            <h3 className="text-base font-medium font-mono uppercase tracking-wider">
              Verified Research Achievements (N=1,250)
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="p-5 rounded-xl bg-surface-200/80 border border-border/70">
              <span className="text-xs font-mono text-accent-teal font-semibold block mb-1">
                91.4% Relative Reduction in Hallucinations
              </span>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                Slashed unsupported generated claims from 37.08% down to 3.20% through evidence verification gating.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-surface-200/80 border border-border/70">
              <span className="text-xs font-mono text-accent-teal font-semibold block mb-1">
                94.50% Attribution Coverage
              </span>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                Established verifiable sentence-level citation anchors [1], [2] pointing to retrieved context passages.
              </p>
            </div>

            <div className="p-5 rounded-xl bg-surface-200/80 border border-border/70">
              <span className="text-xs font-mono text-accent-teal font-semibold block mb-1">
                71.60% Safe Abstention
              </span>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                Correctly refused to answer 358 out of 500 missing or contradictory questions (vs 0.0% in Standard RAG).
              </p>
            </div>

            <div className="p-5 rounded-xl bg-surface-200/80 border border-border/70">
              <span className="text-xs font-mono text-accent-teal font-semibold block mb-1">
                72.40% GPU Compute Savings
              </span>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                Avoided 905 expensive GPU generation calls through early verifier exit, accelerating average latency by 70.7%.
              </p>
            </div>
          </div>

          <div className="mt-8 pt-4 border-t border-border/60 flex items-center justify-between text-xs font-mono text-foreground-muted">
            <span>Validated via paired McNemar testing (p = 1.01 × 10⁻¹⁴)</span>
            <span className="text-accent-teal">100% Empirically Verified</span>
          </div>
        </div>
      </div>
    </section>
  );
};
