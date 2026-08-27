import React from 'react';
import { Sparkles, Layers, ArrowRight, ShieldCheck } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';
import { TableHeaderTooltip } from '@/components/TableHeaderTooltip';

export const GenerationSection: React.FC = () => {
  const { generation_ablation_experiments } = CANONICAL_RESEARCH_DATA;

  return (
    <section className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            08 / Controlled Generation Experiments
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Ablation Experiments: G-A through G-F
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            To evaluate generation strategies independently from retrieval and verification, controlled experiments G-A through G-F were executed across the verified evidence subset.
          </p>
        </div>

        {/* 6 Ablation Table */}
        <div className="overflow-x-auto mb-8 overflow-visible">
          <table className="w-full text-left border-collapse border border-border rounded-xl bg-surface-100 font-sans text-xs">
            <thead>
              <tr className="bg-surface-200 border-b border-border text-[11px] font-mono text-foreground-muted uppercase tracking-wider">
                <th className="p-4 relative overflow-visible">
                  <TableHeaderTooltip
                    label="Exp ID"
                    calculation="Identifier for the generation prompt ablation experiment."
                  />
                </th>
                <th className="p-4 relative overflow-visible">
                  <TableHeaderTooltip
                    label="Strategy"
                    calculation="The specific prompt template and attribution constraint applied to the LLM."
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Exact Match"
                    calculation="Percentage of answers matching gold standard reference string exactly."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Token F1"
                    calculation="Word overlap precision and recall harmonic mean against reference string."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Supported Claim %"
                    calculation="Percentage of generated sentences with verifiable evidence grounding."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Unsupported %"
                    calculation="Percentage of generated statements that cannot be verified in evidence."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Attribution %"
                    calculation="Percentage of sentences containing valid bracketed citations [1], [2]."
                    align="right"
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Faithfulness %"
                    calculation="Percentage of generated statements faithful to context without contradiction."
                    align="right"
                  />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {generation_ablation_experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-surface-50/50 transition-colors">
                  <td className="p-4 font-mono font-medium text-accent-teal">{exp.id}</td>
                  <td className="p-4">
                    <span className="font-medium text-foreground block">{exp.name}</span>
                    <span className="text-[11px] text-foreground-muted block mt-0.5">{exp.strategy}</span>
                  </td>
                  <td className="p-4 font-mono text-right text-foreground">{exp.exact_match.toFixed(2)}%</td>
                  <td className="p-4 font-mono text-right text-foreground">{exp.token_f1.toFixed(4)}</td>
                  <td className="p-4 font-mono text-right text-accent-teal">{exp.supported_claim_rate.toFixed(2)}%</td>
                  <td className="p-4 font-mono text-right text-accent-coral">{exp.unsupported_claim_rate.toFixed(2)}%</td>
                  <td className="p-4 font-mono text-right text-foreground">{exp.attribution_coverage.toFixed(2)}%</td>
                  <td className="p-4 font-mono text-right text-foreground">{exp.faithfulness_score.toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Methodological Isolation Callout */}
        <div className="p-4 rounded-xl bg-surface-200/50 border border-border/60 text-xs font-sans text-foreground-muted flex items-center justify-between">
          <span>
            These values represent the controlled generation prompt ablation (Exp G-A to G-F) on the permitted answer population.
          </span>
          <span className="text-xs font-mono text-accent-teal shrink-0 ml-4">
            Controlled Prompt Ablations
          </span>
        </div>
      </div>
    </section>
  );
};
