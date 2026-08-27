import React, { useState } from 'react';
import { FileText, CheckCircle2, XCircle, Search, ExternalLink, HelpCircle, Info } from 'lucide-react';
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
  const [hoveredScore, setHoveredScore] = useState<string | null>(null);

  return (
    <div id="evidence-inspector" className="p-7 rounded-2xl bg-surface-100 border border-border mb-12 scroll-mt-24">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 pb-4 border-b border-border/60">
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

      {/* Score Explanation Banner */}
      <div className="p-3.5 rounded-xl bg-surface-200/60 border border-border/60 flex items-start gap-2.5 text-xs font-sans text-foreground-muted mb-6">
        <Info className="w-4 h-4 text-accent-teal shrink-0 mt-0.5" />
        <div>
          <strong className="text-foreground font-mono text-[11px] block mb-0.5">
            How Passage Relevance Scores Are Calculated:
          </strong>
          <span>
            Each score represents the <strong>Cosine Similarity</strong> <code className="text-foreground font-mono">cos(q, d)</code> between the 384-dimensional question vector and the document vector using the <code className="text-foreground font-mono">BAAI/bge-small-en-v1.5</code> dense embedding model. Scores range from 0.0 (unrelated) to 1.0 (exact semantic match).
          </span>
        </div>
      </div>

      {/* Chunks List */}
      <div className="space-y-4">
        {chunks.map((chunk) => {
          const isFocused = activeChunkNumber === chunk.chunkNumber;
          return (
            <div
              key={chunk.id}
              onClick={() => onSelectChunk(isFocused ? null : chunk.chunkNumber)}
              className={`p-5 rounded-xl border transition-all cursor-pointer relative ${
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
                  {/* Score Pill with Hover Tooltip */}
                  <div
                    className="relative"
                    onMouseEnter={() => setHoveredScore(chunk.id)}
                    onMouseLeave={() => setHoveredScore(null)}
                  >
                    <span className="text-foreground font-semibold px-2 py-0.5 rounded bg-surface-300 border border-border/80 cursor-help flex items-center gap-1">
                      Score: {chunk.score.toFixed(3)}
                      <HelpCircle className="w-3 h-3 text-foreground-muted" />
                    </span>

                    {hoveredScore === chunk.id && (
                      <div className="absolute right-0 top-full mt-1.5 z-50 w-64 p-2.5 rounded-lg bg-surface-100 border border-border shadow-2xl text-[11px] font-sans text-foreground-muted pointer-events-none">
                        <strong className="text-foreground block font-mono text-[10px] uppercase mb-0.5">
                          BGE Cosine Similarity
                        </strong>
                        Dot product of normalized question and passage embeddings. A score of {chunk.score.toFixed(3)} indicates high semantic topical relevance.
                      </div>
                    )}
                  </div>

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
