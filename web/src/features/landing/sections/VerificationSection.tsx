import React from 'react';
import { ShieldCheck, AlertTriangle, CheckCircle2, XOctagon } from 'lucide-react';

export const VerificationSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            06 / Verification Layer
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Evidence Verification: <br />
            <span className="italic text-accent-teal">44.80% Calibrated Accuracy</span>
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            While Standard RAG contains no verification layer at all (passing retrieved context blindly to the generator), ClearRAG introduces an explicit verification stage that examines retrieved chunks before deciding whether to answer.
          </p>
        </div>

        {/* Verification Overview Feature Card */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-accent-teal/40 bg-accent-teal/5 mb-12">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-border/60">
            <div>
              <span className="text-xs font-mono uppercase text-accent-teal tracking-wider block mb-1">
                ClearRAG Verification Engine
              </span>
              <h3 className="text-xl font-medium text-foreground">
                Semantic & Contradiction Verification (44.80% Accuracy)
              </h3>
            </div>
            <div className="text-left md:text-right">
              <span className="text-xs font-mono text-foreground-muted block">Standard RAG Verification:</span>
              <span className="text-sm font-mono text-accent-coral font-semibold">0.00% (No Verifier)</span>
            </div>
          </div>

          <p className="text-xs text-foreground-muted font-sans leading-relaxed mt-6 mb-8">
            ClearRAG evaluates every retrieved evidence passage against the query claims using multi-hop entity alignment, non-stopword token overlap, and attribute-level contradiction checks.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-sans text-xs">
            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <strong className="text-foreground font-mono block text-[11px] uppercase mb-1">
                1. Semantic Embedding Match
              </strong>
              <p className="text-foreground-muted leading-relaxed">
                Calculates BGE cosine similarity (<code className="text-foreground">θ ≥ 0.65</code>) combined with non-stopword token overlap (<code className="text-foreground">overlap ≥ 0.35</code>).
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <strong className="text-foreground font-mono block text-[11px] uppercase mb-1">
                2. Multi-Hop Relation Support
              </strong>
              <p className="text-foreground-muted leading-relaxed">
                Requires both intermediate entity bridge passages to be present in context before marking a query as fully supported.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-200/80 border border-border/60">
              <strong className="text-foreground font-mono block text-[11px] uppercase mb-1">
                3. Contradiction Detection
              </strong>
              <p className="text-foreground-muted leading-relaxed">
                Identifies date mismatches and numerical contradictions across passages, routing divergent claims to conflict preservation.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
