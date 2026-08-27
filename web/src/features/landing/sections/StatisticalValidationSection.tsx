import React from 'react';
import { BarChart3, CheckCircle, HelpCircle, ShieldCheck, Layers, Info } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';

export const StatisticalValidationSection: React.FC = () => {
  const { statistical_validation } = CANONICAL_RESEARCH_DATA;
  const { mcnemar, bootstrap_95_ci } = statistical_validation;

  return (
    <section id="statistics" className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        {/* Section Header */}
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            15 / Statistical Significance & Validation
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Statistical Validation (N=1,250)
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            Because Standard RAG and ClearRAG were tested on the exact same 1,250 benchmark queries, paired statistical tests were conducted to confirm that ClearRAG’s safety improvements are real and statistically significant.
          </p>
        </div>

        {/* Plain-English Explanation Guide Box */}
        <div className="p-6 rounded-2xl bg-surface-100 border border-border mb-10">
          <div className="flex items-center gap-2 mb-4 text-accent-teal">
            <Info className="w-4 h-4" />
            <h3 className="text-xs font-mono uppercase tracking-wider font-semibold">
              Understanding the Statistical Terms
            </h3>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-sans text-foreground-muted">
            <div className="p-3.5 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                What is McNemar's Paired Test?
              </strong>
              <span>
                A statistical test designed for comparing two AI models tested on the <em>exact same questions</em>. It counts how often ClearRAG made a safe decision when Standard RAG failed (and vice versa) to prove the difference is not a coincidence.
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                What is a p-value (p &lt; 0.001)?
              </strong>
              <span>
                The probability that these results occurred purely by luck or random chance. A value below 0.05 is considered significant; our value (<code className="text-foreground">p = 1.01 × 10⁻¹⁴</code>) means there is virtually <strong>zero chance</strong> this was a fluke.
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                What is Chi-Squared (χ² = 59.87)?
              </strong>
              <span>
                A score measuring how far ClearRAG’s safety choices diverged from Standard RAG. A higher score means a massive, decisive shift toward safe abstention on unanswerable questions.
              </span>
            </div>

            <div className="p-3.5 rounded-xl bg-surface-200/70 border border-border/60">
              <strong className="text-foreground font-medium block mb-1">
                What is the Odds Ratio (1.93x)?
              </strong>
              <span>
                ClearRAG is <strong>1.93 times more likely</strong> to make a correct or safe decision on discordant benchmark queries compared to Standard RAG (397 wins vs 206 for Standard RAG).
              </span>
            </div>
          </div>
        </div>

        {/* McNemar's Test Score Card */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-accent-teal/40 bg-accent-teal/5 mb-10">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-mono uppercase text-accent-teal">
              Hypothesis Test Result
            </span>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-accent-teal/20 text-accent-teal border border-accent-teal/30 font-medium">
              p = 1.01 × 10⁻¹⁴ (Statistically Significant)
            </span>
          </div>

          <h3 className="text-xl font-medium text-foreground mb-4">
            {mcnemar.test_name}
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs text-foreground-muted mb-6">
            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <span className="text-foreground-subtle block text-[10px] uppercase mb-1">p-value</span>
              <span className="text-accent-teal font-semibold text-sm">{mcnemar.p_value}</span>
            </div>
            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <span className="text-foreground-subtle block text-[10px] uppercase mb-1">Chi-squared (χ²)</span>
              <span className="text-foreground font-semibold text-sm">{mcnemar.chi2}</span>
            </div>
            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <span className="text-foreground-subtle block text-[10px] uppercase mb-1">Odds Ratio</span>
              <span className="text-foreground font-semibold text-sm">{mcnemar.odds_ratio}x</span>
            </div>
          </div>

          <p className="text-xs text-foreground-muted font-sans leading-relaxed">
            {mcnemar.interpretation}
          </p>
        </div>

        {/* 1,000 Bootstrap Resamples Table & Explanation */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-border">
          <div className="mb-6">
            <div className="mb-2">
              <span className="text-xs font-mono uppercase tracking-wider font-semibold text-foreground">
                Bootstrap 95% Confidence Intervals (1,000 Resamples)
              </span>
            </div>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-3">
              To test whether our results remain stable across random subsets, a computer algorithm repeatedly resampled 1,000 randomized subsets of questions.
            </p>

          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-sans text-xs border-collapse">
              <thead>
                <tr className="border-b border-border/60 text-foreground-muted font-mono text-[11px]">
                  <th className="py-3">Metric Dimension</th>
                  <th className="py-3 text-accent-teal">ClearRAG 95% CI [Lower, Upper]</th>
                  <th className="py-3 text-foreground-muted">Standard RAG 95% CI [Lower, Upper]</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40 font-mono">
                {bootstrap_95_ci.metrics.map((m) => (
                  <tr key={m.name} className="hover:bg-surface-50/40 transition-colors">
                    <td className="py-3 font-sans text-foreground font-medium">{m.name}</td>
                    <td className="py-3 text-accent-teal font-medium">{m.clearrag_ci}</td>
                    <td className="py-3 text-foreground-muted">{m.standard_rag_ci}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
};
