import React from 'react';
import { BookOpen, CheckCircle2, ShieldCheck, Link2 } from 'lucide-react';

export const AttributionSection: React.FC = () => {
  return (
    <section className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            09 / Claim-Level Attribution
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            94.50% Verifiable Attribution Coverage
          </h2>
          <p className="max-w-2xl text-base text-foreground-muted font-sans font-light leading-relaxed">
            ClearRAG enforces sentence-level citation anchors, mapping every generated claim to verifiable evidence passages. While Standard RAG produces zero explicit citations, ClearRAG achieves 94.50% attribution coverage.
          </p>
        </div>

        {/* Visual Citation Mapping Demo Card */}
        <div className="p-8 rounded-2xl bg-surface-100 border border-border mb-10">
          <span className="text-xs font-mono uppercase text-foreground-muted block mb-4">
            Example: Sentence-Level Evidence Alignment
          </span>

          <div className="space-y-6">
            {/* Generated Answer with Citations */}
            <div className="p-4 rounded-xl bg-surface-200 border border-border text-sm text-foreground font-sans leading-relaxed">
              “Walter Hill directed the western film The Long Riders in 1980 <span className="font-mono text-xs px-1.5 py-0.5 rounded bg-accent-teal/20 text-accent-teal border border-accent-teal/30 cursor-pointer font-bold">[1]</span>, and later directed The Driver in 1978 <span className="font-mono text-xs px-1.5 py-0.5 rounded bg-accent-teal/20 text-accent-teal border border-accent-teal/30 cursor-pointer font-bold">[2]</span>. Born in 1942, Hill was 38 years old when The Long Riders was released <span className="font-mono text-xs px-1.5 py-0.5 rounded bg-accent-teal/20 text-accent-teal border border-accent-teal/30 cursor-pointer font-bold">[1]</span>.”
            </div>

            {/* Evidence Passages */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-4 rounded-xl bg-surface-300 border border-accent-teal/30">
                <div className="flex items-center justify-between text-accent-teal mb-2">
                  <span>Evidence Chunk [1]</span>
                  <span className="text-[10px] text-foreground-subtle">Passage #0842</span>
                </div>
                <p className="text-foreground-muted font-sans leading-relaxed">
                  “The Long Riders is a 1980 American Western film directed by Walter Hill (born January 10, 1942)...”
                </p>
              </div>

              <div className="p-4 rounded-xl bg-surface-300 border border-accent-teal/30">
                <div className="flex items-center justify-between text-accent-teal mb-2">
                  <span>Evidence Chunk [2]</span>
                  <span className="text-[10px] text-foreground-subtle">Passage #0913</span>
                </div>
                <p className="text-foreground-muted font-sans leading-relaxed">
                  “The Driver is a 1978 crime thriller film written and directed by Walter Hill...”
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Formula Card */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="p-6 rounded-xl bg-surface-100 border border-border">
            <span className="text-xs font-mono uppercase text-foreground-muted block mb-2">
              Attribution Coverage Formula
            </span>
            <div className="text-xs font-mono text-foreground p-3 rounded bg-surface-200 border border-border/60 mb-2">
              Attribution Coverage = (Claims with Valid Citations / Total Claims) × 100%
            </div>
            <p className="text-xs text-foreground-muted font-sans">
              Measures how many generated assertions include verifiable evidence links.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-surface-100 border border-border">
            <span className="text-xs font-mono uppercase text-foreground-muted block mb-2">
              Attribution Precision
            </span>
            <div className="text-xs font-mono text-foreground p-3 rounded bg-surface-200 border border-border/60 mb-2">
              Attribution Precision = 95.20%
            </div>
            <p className="text-xs text-foreground-muted font-sans">
              95.2% of generated bracketed citation links accurately match the supporting evidence.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};
