import React from 'react';
import { AlertCircle, GitMerge, AlertTriangle, ShieldCheck } from 'lucide-react';

export const CaveatConflictSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-gold mb-3">
            10 / Caveats & Conflict Handling
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Partial Evidence & Contradictions
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            Real-world knowledge bases contain partial facts and contradictory sources. Rather than hallucinating missing links or arbitrarily picking a side, ClearRAG preserves transparency.
          </p>
        </div>

        {/* 2 Big Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Card 1: Caveat-Aware Synthesis */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-accent-cyan tracking-wider">
                  Caveat-Aware Generation
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20">
                  98.40% Compliance
                </span>
              </div>
              <h3 className="text-lg font-medium text-foreground mb-3">
                Transparent Partial Evidence Handling
              </h3>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-6">
                When only one entity of a two-entity comparison question is retrieved, ClearRAG does not guess the missing half. It answers what is verified and prefixes a structured caveat.
              </p>

              <div className="p-4 rounded-xl bg-surface-200 border border-border text-xs font-sans text-foreground-muted">
                <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-1">Generated Output Format</span>
                <p className="italic text-foreground">
                  “<strong className="text-accent-cyan font-mono not-italic text-[11px]">[Caveat: Partial evidence available]</strong> Bactris contains approximately 79 species [1]. Information regarding the species count of the second target was not found in retrieved context.”
                </p>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-border/60 text-xs font-mono text-foreground-subtle">
              Caveat Compliance: 98.40% of partial evidence answers correctly included caveat hedging.
            </div>
          </div>

          {/* Card 2: Conflict-Aware Handling */}
          <div className="p-7 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-xs font-mono uppercase text-accent-amber tracking-wider">
                  Conflict Preservation
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-accent-amber/10 text-accent-amber border border-accent-amber/20">
                  76.80% Preserved
                </span>
              </div>
              <h3 className="text-lg font-medium text-foreground mb-3">
                Avoiding Arbitrary Disagreement Resolution
              </h3>
              <p className="text-xs text-foreground-muted font-sans leading-relaxed mb-6">
                When retrieved passages contain conflicting facts (e.g. birth year 1904 vs 1907), Standard RAG arbitrarily outputs one number. ClearRAG preserves both perspectives or triggers a conflict abstention.
              </p>

              <div className="p-4 rounded-xl bg-surface-200 border border-border text-xs font-sans text-foreground-muted">
                <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-1">Conflict Detection Response</span>
                <p className="italic text-foreground">
                  “<strong className="text-accent-amber font-mono not-italic text-[11px]">[Conflicting Evidence Detected]</strong> Retrieved sources disagree regarding Thomas Carr’s birth year: Source [1] reports 1904, whereas Source [2] reports 1907.”
                </p>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-border/60 text-xs font-mono text-foreground-subtle">
              Safe Handling: 192 of 250 conflicting queries safely identified and preserved.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
