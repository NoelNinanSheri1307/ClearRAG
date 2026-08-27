import React from 'react';
import { FileText, CheckCircle2, XCircle, Search, ExternalLink } from 'lucide-react';
import { EvidenceChunk } from '@/data/demo_scenarios';

interface EvidenceInspectorProps {
  chunks: EvidenceChunk[];
  activeChunkNumber: number | null;
  onSelectChunk: (num: number | null) => void;
}

export const EvidenceInspector: React.FC<EvidenceInspectorProps> = ({
  chunks,
  activeChunkNumber,
  onSelectChunk,
}) => {
  return (
    <div id="evidence-inspector" className="p-7 rounded-2xl bg-surface-100 border border-border mb-12 scroll-mt-24">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6 pb-4 border-b border-border/60">
        <div>
          <span className="text-[10px] font-mono uppercase text-accent-teal tracking-wider block mb-0.5">
            Step 4 • Retrieved Evidence Inspector
          </span>
          <h3 className="text-base font-medium text-foreground">
            Source Context Chunks & Claim Grounding
          </h3>
        </div>
        <span className="text-xs font-mono text-foreground-muted">
          {chunks.length} Retrieved Wikipedia Passages
        </span>
      </div>

      <div className="space-y-4">
        {chunks.map((chunk) => {
          const isFocused = activeChunkNumber === chunk.chunkNumber;
          return (
            <div
              key={chunk.id}
              onClick={() => onSelectChunk(isFocused ? null : chunk.chunkNumber)}
              className={`p-5 rounded-xl border transition-all cursor-pointer ${
                isFocused
                  ? 'bg-accent-teal/15 border-accent-teal shadow-xl ring-2 ring-accent-teal animate-pulse'
                  : 'bg-surface-200/60 border-border/70 hover:border-border hover:bg-surface-200/90'
              }`}
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                <div className="flex items-center gap-2.5">
                  <span className={`text-xs font-mono px-2 py-0.5 rounded font-bold ${
                    isFocused ? 'bg-accent-teal text-background' : 'bg-surface-300 text-accent-teal'
                  }`}>
                    [{chunk.chunkNumber}]
                  </span>
                  <strong className="text-xs font-medium text-foreground">
                    {chunk.title}
                  </strong>
                </div>

                <div className="flex items-center gap-2 font-mono text-[11px]">
                  <span className="text-foreground-subtle">
                    Score: {chunk.score.toFixed(3)}
                  </span>
                  <span className={`px-2 py-0.5 rounded border text-[10px] uppercase font-medium ${
                    chunk.isSupporting
                      ? 'bg-accent-teal/15 text-accent-teal border-accent-teal/30'
                      : chunk.sourceType === 'synthetic_conflict'
                      ? 'bg-accent-coral/15 text-accent-coral border-accent-coral/30'
                      : 'bg-surface-300 text-foreground-muted border-border'
                  }`}>
                    {chunk.isSupporting ? 'Supporting Evidence' : chunk.sourceType === 'synthetic_conflict' ? 'Contradictory Claim' : 'Distractor'}
                  </span>
                </div>
              </div>

              <p className="text-xs font-sans text-foreground-muted leading-relaxed">
                "{chunk.text}"
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};
