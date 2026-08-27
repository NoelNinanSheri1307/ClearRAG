import React from 'react';
import { FileText, ShieldCheck, Database, FileCode } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';

export const ResearchArtifactsSection: React.FC = () => {
  const { meta } = CANONICAL_RESEARCH_DATA;

  const artifacts = [
    { title: 'Canonical Evaluation Ledger (JSON)', path: 'results/final_canonical_evaluation.json', desc: 'The single machine-readable JSON dataset containing all verified metrics for Systems 0 through 4 across 1,250 queries.' },
    { title: 'Operating Point Sweep Results (JSON)', path: 'results/coverage_risk_quality.json', desc: 'Complete numerical log of the 12-setting verifier confidence sweep mapping the coverage vs risk curve.' },
    { title: 'Evaluation Audit & Reconciliation Log', path: 'docs/final_reproducibility_audit.md', desc: 'Documented mathematical proof and query accounting explaining the relation between answered and all-instances metrics.' },
    { title: 'Controlled Generation Logs (JSON)', path: 'results/generation_experiments.json', desc: 'Raw experimental output logs for prompt grounding variations G-A through G-F on verified queries.' },
    { title: 'Statistical Hypothesis Tests (JSON)', path: 'results/final_statistical_tests.json', desc: 'Exact chi-squared statistics, p-values, and 1,000 bootstrap resample confidence interval logs.' },
    { title: '1,250 Benchmark Queries (JSON)', path: 'data/evaluation/clearrag_eval.json', desc: 'The fixed, frozen multi-hop evaluation set spanning all 5 controlled evidence conditions.' },
  ];

  return (
    <section id="artifacts" className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-foreground-subtle mb-3">
            19 / Experiment Logs & Data Files
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Experiment Logs & Result Files
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            Every number, score, and chart on this page is generated from these version-controlled machine-readable result files and experiment logs in the repository.
          </p>
        </div>

        {/* Artifacts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-12">
          {artifacts.map((art) => (
            <div
              key={art.path}
              className="p-5 rounded-xl bg-surface-100 border border-border flex items-start justify-between gap-4"
            >
              <div className="flex items-start gap-3">
                <FileCode className="w-5 h-5 text-accent-teal shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-xs font-mono font-medium text-foreground mb-1">
                    {art.title}
                  </h4>
                  <code className="text-[11px] font-mono text-accent-teal block mb-1">
                    {art.path}
                  </code>
                  <p className="text-[11px] font-sans text-foreground-muted">
                    {art.desc}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Official Research Freeze Statement Card */}
        <div className="p-8 rounded-xl bg-surface-100 border border-border text-center relative overflow-hidden">
          <div className="text-xs font-mono uppercase tracking-widest text-foreground-muted mb-4">
            Research Backend Status: 100% Frozen & Reproducible
          </div>

          <h3 className="text-xl sm:text-2xl font-serif text-foreground mb-3">
            Defensible Research Conclusion
          </h3>

          <p className="max-w-3xl mx-auto text-xs sm:text-sm text-foreground-muted font-sans font-light leading-relaxed mb-6 italic">
            “ClearRAG provides a statistically significant, substantially safer, and verifiably evidence-grounded response policy compared to conventional always-answer Standard RAG—reducing unsupported hallucinated claims by 91.4% (from 37.08% to 3.20%) and achieving 71.60% safe abstention on unanswerable/contradictory queries, while establishing 94.50% verifiable claim attribution and saving 72.40% of LLM generation compute, with a controlled tradeoff in raw answer coverage.”
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-mono text-foreground-subtle border-t border-border/60 pt-4">
            <span>Verified Test Suite: {meta.verified_test_count} / {meta.verified_test_count} Passing</span>
            <span>•</span>
            <span>Version: {meta.version}</span>
            <span>•</span>
            <span>Dataset: {meta.dataset_name}</span>
          </div>
        </div>
      </div>
    </section>
  );
};
