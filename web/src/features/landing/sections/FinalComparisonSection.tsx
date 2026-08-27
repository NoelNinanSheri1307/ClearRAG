import React from 'react';
import { ShieldAlert, ShieldCheck, ArrowRight, Minus, Check, X } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';
import { TableHeaderTooltip } from '@/components/TableHeaderTooltip';

export const FinalComparisonSection: React.FC = () => {
  const { canonical_systems } = CANONICAL_RESEARCH_DATA;
  const sys0 = canonical_systems.system_0;
  const sys4 = canonical_systems.system_4;

  const comparisonRows = [
    { 
      label: 'Answer Coverage Rate', 
      std: `${sys0.answer_coverage_rate.toFixed(2)}% (1,250 / 1,250)`, 
      clear: `${sys4.answer_coverage_rate.toFixed(2)}% (345 / 1,250)`, 
      delta: '-72.40 pp', 
      note: 'Percentage of benchmark queries where an answer was generated',
      calculation: 'Count of generated answers divided by total 1,250 benchmark queries.'
    },
    { 
      label: 'Unsupported Claim Rate (Hallucinations)', 
      std: `${sys0.unsupported_claim_rate.toFixed(2)}%`, 
      clear: `${sys4.unsupported_claim_rate.toFixed(2)}%`, 
      delta: '-91.4% Relative', 
      note: 'Percentage of generated factual statements with no evidence',
      calculation: 'Count of unsupported sentences divided by total generated sentences in answers.',
      highlightClear: true 
    },
    { 
      label: 'Supported Claim Rate', 
      std: `${sys0.supported_claim_rate.toFixed(2)}%`, 
      clear: `${sys4.supported_claim_rate.toFixed(2)}%`, 
      delta: '+33.88 pp', 
      note: 'Percentage of factual claims backed by retrieved context',
      calculation: '100% minus the Unsupported Claim Rate.',
      highlightClear: true 
    },
    { 
      label: 'Attribution Coverage', 
      std: `${sys0.attribution_coverage.toFixed(2)}%`, 
      clear: `${sys4.attribution_coverage.toFixed(2)}%`, 
      delta: '+94.50 pp', 
      note: 'Generated sentences that include verifiable citation markers',
      calculation: 'Percentage of sentences that include valid bracketed citations [1], [2] aligned to context.',
      highlightClear: true 
    },
    { 
      label: 'Safe Abstention on Unanswerables', 
      std: `${sys0.safe_abstention_rate.toFixed(2)}% (0 / 500)`, 
      clear: `${sys4.safe_abstention_rate.toFixed(2)}% (358 / 500)`, 
      delta: '+71.60 pp', 
      note: 'Correct refusal when evidence is missing or contradictory',
      calculation: 'Count of correct abstentions on unanswerable/conflict queries divided by 500 unanswerable queries.',
      highlightClear: true 
    },
    { 
      label: 'Unsafe Answer Rate on Unanswerables', 
      std: `${sys0.unsafe_answer_rate.toFixed(2)}% (500 / 500)`, 
      clear: `${sys4.unsafe_answer_rate.toFixed(2)}% (142 / 500)`, 
      delta: '-71.60 pp', 
      note: 'Dangerous guessing on queries that have no supporting facts',
      calculation: 'Percentage of unanswerable queries where the model erroneously generated an answer.',
      highlightClear: true 
    },
    { 
      label: 'Answered-Instance Exact Match (EM)', 
      std: `${sys0.generated_exact_match.toFixed(2)}%`, 
      clear: `${sys4.generated_exact_match.toFixed(2)}%`, 
      delta: '-5.01 pp', 
      note: 'String match against gold reference only on answered queries',
      calculation: 'Normalized string exact match scored only over queries where the model actually produced an answer.',
      highlightStd: true 
    },
    { 
      label: 'Answered-Instance Token F1', 
      std: `${sys0.generated_token_f1.toFixed(4)}`, 
      clear: `${sys4.generated_token_f1.toFixed(4)}`, 
      delta: '-0.0893', 
      note: 'Token overlap against gold answer only on answered queries',
      calculation: 'Harmonic mean of precision and recall of shared words between answer and reference, on answered queries.',
      highlightStd: true 
    },
    { 
      label: 'All-Instances Exact Match (EM)', 
      std: `${sys0.all_instances_exact_match.toFixed(2)}%`, 
      clear: `${sys4.all_instances_exact_match.toFixed(2)}%`, 
      delta: '-9.84 pp', 
      note: 'Exact match where every abstained query receives score 0.0',
      calculation: 'Answered Exact Match multiplied by Answer Coverage Rate (abstained queries count as 0.0).',
      highlightStd: true 
    },
    { 
      label: 'All-Instances Token F1', 
      std: `${sys0.all_instances_token_f1.toFixed(4)}`, 
      clear: `${sys4.all_instances_token_f1.toFixed(4)}`, 
      delta: '-0.2113', 
      note: 'Token F1 where every abstained query receives score 0.0',
      calculation: 'Answered Token F1 multiplied by Answer Coverage Rate (abstained queries count as 0.0).',
      highlightStd: true 
    },
    { 
      label: 'LLM Generation Calls Avoided', 
      std: '0 calls (0.0%)', 
      clear: '905 calls (72.40%)', 
      delta: '+72.40% Saved', 
      note: 'GPU inference calls skipped due to early verifier abstention',
      calculation: 'Count of queries where decision layer avoided invoking the generator LLM.',
      highlightClear: true 
    },
    { 
      label: 'Mean Pipeline Latency', 
      std: `${sys0.mean_total_latency_ms.toFixed(1)} ms`, 
      clear: `${sys4.mean_total_latency_ms.toFixed(1)} ms`, 
      delta: '-70.7% Faster', 
      note: 'Total end-to-end execution time per query',
      calculation: 'Average time in milliseconds from receiving query to final output across all 1,250 queries.',
      highlightClear: true 
    },
  ];

  return (
    <section id="comparison" className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            11 / Final System Comparison
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Standard RAG vs ClearRAG
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            The canonical comparative evaluation across all 1,250 benchmark queries under identical evaluation code and zero metadata leakage. Hover over any column header to see simple calculation rules.
          </p>
        </div>

        {/* Big Canonical Table */}
        <div className="overflow-x-auto mb-10 overflow-visible">
          <table className="w-full text-left border-collapse border border-border rounded-xl bg-surface-100 font-sans text-xs">
            <thead>
              <tr className="bg-surface-200 border-b border-border text-[11px] font-mono uppercase tracking-wider text-foreground-muted">
                <th className="p-4 relative overflow-visible">
                  <TableHeaderTooltip
                    label="Dimension / Metric"
                    calculation="The evaluation axis being tested across the 1,250 benchmark questions."
                  />
                </th>
                <th className="p-4 relative overflow-visible">
                  <TableHeaderTooltip
                    label="Standard RAG Baseline"
                    calculation="Linear retrieve-then-generate baseline using BGE embeddings (k=5) + unconstrained Qwen 1.5B LLM."
                  />
                </th>
                <th className="p-4 text-accent-teal relative overflow-visible">
                  <TableHeaderTooltip
                    label="ClearRAG"
                    calculation="Full ClearRAG pipeline with Hybrid RRF retrieval (k=10), semantic verification, calibrated decision gating, and grounded attribution."
                  />
                </th>
                <th className="p-4 text-right relative overflow-visible">
                  <TableHeaderTooltip
                    label="Delta / Impact"
                    calculation="The direct numerical difference (percentage points or relative reduction) between ClearRAG and Standard RAG."
                    align="right"
                  />
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 font-mono">
              {comparisonRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-surface-50/50 transition-colors">
                  <td className="p-4 font-sans font-medium text-foreground">
                    {row.label}
                    <span className="block text-[11px] font-mono text-foreground-muted font-normal mt-0.5">{row.note}</span>
                  </td>
                  <td className={`p-4 ${row.highlightStd ? 'text-foreground font-semibold' : 'text-foreground-muted'}`}>
                    {row.std}
                  </td>
                  <td className={`p-4 ${row.highlightClear ? 'text-accent-teal font-semibold' : 'text-foreground'}`}>
                    {row.clear}
                  </td>
                  <td className={`p-4 text-right font-medium ${row.highlightClear ? 'text-accent-teal' : row.highlightStd ? 'text-foreground-muted' : 'text-foreground'}`}>
                    {row.delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="p-4 rounded-xl bg-surface-200/50 border border-border/60 text-xs font-mono text-foreground-muted flex items-center justify-between">
          <span>N=1,250 Controlled Benchmark</span>
          <span className="text-accent-teal">Identical Evaluation Code & Zero Leakage</span>
        </div>
      </div>
    </section>
  );
};
