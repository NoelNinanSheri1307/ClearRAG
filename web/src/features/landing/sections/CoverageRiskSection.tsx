import React from 'react';
import { Layers, Sliders, TrendingUp, ShieldAlert, Sparkles, HelpCircle } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';
import { TableHeaderTooltip } from '@/components/TableHeaderTooltip';

export const CoverageRiskSection: React.FC = () => {
  const { operating_points_sweep } = CANONICAL_RESEARCH_DATA;

  return (
    <section className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-10">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-cyan mb-3">
            13 / Configurable Operating Settings
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Tunable Decision Settings & Operating Points
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed mb-6">
            An <strong>Operating Point</strong> is a configured threshold setting for the verifier. By tuning the confidence threshold (<code className="text-foreground">θ</code>), you choose how strict or permissive ClearRAG is when verifying evidence before generating an answer.
          </p>

          <div className="p-4 rounded-xl bg-surface-100 border border-border flex items-start gap-3 text-xs font-sans text-foreground-muted">
            <HelpCircle className="w-4 h-4 text-accent-cyan shrink-0 mt-0.5" />
            <span>
              <strong className="text-foreground">What is an Operating Point?</strong> Higher threshold (e.g. θ=0.90) means strict evidence requirement (zero hallucinations, but fewer questions answered). Lower threshold (e.g. θ=0.30) means more questions answered, with slightly more tolerance for imperfect evidence.
            </span>
          </div>
        </div>

        {/* Operating Points Table */}
        <div className="overflow-x-auto mb-10 overflow-visible">
          <table className="w-full text-left border-collapse border border-border rounded-xl bg-surface-100 font-sans text-xs">
            <thead>
              <tr className="bg-surface-200 border-b border-border text-[11px] font-mono uppercase tracking-wider text-foreground-muted">
                <th className="p-4 relative overflow-visible">
                  <TableHeaderTooltip
                    label="Configured Setting"
                    calculation="Named operating configuration representing the verifier strictness profile."
                  />
                </th>
                <th className="p-4 text-center">
                  <TableHeaderTooltip
                    label="Threshold (θ)"
                    calculation="The minimum cosine similarity score required between a claim and an evidence chunk to pass verification."
                    align="center"
                  />
                </th>
                <th className="p-4 text-right">
                  <TableHeaderTooltip
                    label="Coverage %"
                    calculation="Percentage of 1,250 queries that passed verification and produced an answer."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right">
                  <TableHeaderTooltip
                    label="Answered EM %"
                    calculation="Exact match accuracy against the gold answer, calculated only on queries that were answered."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right">
                  <TableHeaderTooltip
                    label="Answered F1"
                    calculation="Token overlap F1 score against gold answer, calculated only on queries that were answered."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right text-accent-coral">
                  <TableHeaderTooltip
                    label="Unsupported %"
                    calculation="Percentage of generated sentences with no factual basis in retrieved context (Hallucinations)."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right">
                  <TableHeaderTooltip
                    label="Unsafe %"
                    calculation="Percentage of unanswerable/conflict queries where the system made an unsafe guess instead of abstaining."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right text-accent-teal">
                  <TableHeaderTooltip
                    label="Attribution %"
                    calculation="Percentage of generated sentences with verified bracketed citation markers [1]."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right">
                  <TableHeaderTooltip
                    label="Compute Saved %"
                    calculation="Percentage of expensive LLM generation calls avoided through early verifier exit."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right">
                  <TableHeaderTooltip
                    label="Utility"
                    calculation="Composite multi-objective score balancing coverage, answer quality, and factual safety."
                    align="right"
                  />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 font-mono">
              {operating_points_sweep.map((op) => (
                <tr key={op.name} className={`hover:bg-surface-50/50 transition-colors ${op.name.includes('Default') ? 'bg-accent-teal/5' : op.name.includes('Max Quality') ? 'bg-accent-cyan/5' : op.name.includes('Balanced') ? 'bg-accent-gold/5' : ''}`}>
                  <td className="p-4 font-sans font-medium text-foreground">
                    {op.name}
                  </td>
                  <td className="p-4 text-center text-foreground-muted">{op.sim_threshold.toFixed(2)}</td>
                  <td className="p-4 text-right text-foreground">{op.answer_coverage_rate.toFixed(1)}%</td>
                  <td className="p-4 text-right text-foreground">{op.answered_exact_match.toFixed(2)}%</td>
                  <td className="p-4 text-right text-foreground">{op.answered_token_f1.toFixed(4)}</td>
                  <td className="p-4 text-right text-accent-coral">{op.unsupported_claim_rate.toFixed(2)}%</td>
                  <td className="p-4 text-right text-foreground-muted">{op.unsafe_answer_rate.toFixed(2)}%</td>
                  <td className="p-4 text-right text-accent-teal">{op.attribution_coverage.toFixed(1)}%</td>
                  <td className="p-4 text-right text-foreground">{op.compute_saved_percentage.toFixed(1)}%</td>
                  <td className="p-4 text-right font-medium text-foreground">{op.composite_utility.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* 3 Key Settings Highlight */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 rounded-2xl bg-surface-100 border border-border">
            <span className="text-[11px] font-mono uppercase text-accent-teal block mb-1">
              Setting 4: Default Calibrated (θ=0.75)
            </span>
            <div className="text-xl font-mono font-medium text-foreground mb-2">
              27.6% Coverage • 3.50% Risk
            </div>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed">
              Maximizes safe abstention on unanswerable queries and saves 72.40% of GPU LLM compute. Recommended for safety-critical enterprise tasks.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-surface-100 border border-accent-gold/40 bg-accent-gold/5">
            <span className="text-[11px] font-mono uppercase text-accent-gold block mb-1">
              Setting 9: Balanced Pareto (θ=0.50)
            </span>
            <div className="text-xl font-mono font-medium text-accent-gold mb-2">
              58.4% Coverage • 9.25% Risk
            </div>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed">
              Highest composite multi-objective score (0.2894). Recovers broad answer volume while maintaining a 4x reduction in hallucinations vs Standard RAG.
            </p>
          </div>

          <div className="p-6 rounded-2xl bg-surface-100 border border-accent-cyan/40 bg-accent-cyan/5">
            <span className="text-[11px] font-mono uppercase text-accent-cyan block mb-1">
              Setting 10: Max Quality (θ=0.45)
            </span>
            <div className="text-xl font-mono font-medium text-accent-cyan mb-2">
              63.8% Coverage • 0.2014 F1
            </div>
            <p className="text-xs text-foreground-muted font-sans leading-relaxed">
              Reaches 7.52% EM and 0.2014 Token F1, recovering 36.84% of the F1 gap against Standard RAG with 85.00% attribution.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};
