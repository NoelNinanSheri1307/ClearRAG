import React from 'react';
import { AlertCircle, CheckCircle2, ShieldCheck, ZapOff, Clock, ShieldAlert, FileText, CornerDownRight } from 'lucide-react';
import { DemoQuestionInstance } from '@/data/demo_scenarios';

interface ComparisonViewProps {
  question: DemoQuestionInstance;
  onCitationClick: (chunkNumber: number) => void;
  activeChunkNumber: number | null;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  question,
  onCitationClick,
  activeChunkNumber,
}) => {
  const { standardRAG, clearRAG } = question;
  const isAbstain = clearRAG.decision.includes('ABSTAIN');

  // Render ClearRAG answer with interactive clickable citation brackets
  const renderAttributedAnswer = (text: string) => {
    const parts = text.split(/(\[\d+\])/g);
    return parts.map((part, index) => {
      const match = part.match(/\[(\d+)\]/);
      if (match) {
        const chunkNum = parseInt(match[1], 10);
        const isActive = activeChunkNumber === chunkNum;
        return (
          <button
            key={index}
            onClick={() => onCitationClick(chunkNum)}
            className={`inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-[11px] font-mono transition-all ${
              isActive
                ? 'bg-accent-teal text-background font-bold shadow-md scale-105'
                : 'bg-accent-teal/20 text-accent-teal hover:bg-accent-teal/30 hover:underline'
            }`}
            title={`Click to focus supporting passage [${chunkNum}] in the Evidence Inspector`}
          >
            {part}
          </button>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="mb-12">
      <div className="flex items-center justify-between mb-4">
        <span className="text-[11px] font-mono uppercase tracking-widest text-foreground-muted">
          Step 3 • Side-by-Side Response Policy Outputs
        </span>
        <span className="text-xs font-mono text-accent-teal">
          Gold Target: "{question.goldAnswer}"
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Standard RAG Column */}
        <div className="p-7 rounded-2xl bg-surface-100 border border-border flex flex-col justify-between">
          <div>
            {/* Header */}
            <div className="flex items-center justify-between mb-4 pb-4 border-b border-border/60">
              <div>
                <span className="text-xs font-mono uppercase text-foreground-muted block mb-0.5">
                  Baseline Control
                </span>
                <h3 className="text-base font-medium text-foreground">
                  Standard RAG
                </h3>
              </div>
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-surface-200 text-foreground-muted border border-border">
                Always-Answer Policy
              </span>
            </div>

            {/* Response Box */}
            <div className="p-5 rounded-xl bg-surface-200/70 border border-border/70 mb-6">
              <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-2">
                Generated Output
              </span>
              <p className="text-sm font-sans text-foreground leading-relaxed">
                {standardRAG.answer}
              </p>
            </div>

            {/* Evidence & Grounding Diagnostics */}
            <div className="space-y-3 font-mono text-xs mb-6">
              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-border/60">
                <span className="text-foreground-muted font-sans">Evidence Support:</span>
                <span className={`font-semibold ${
                  standardRAG.evidenceSupport === 'SUPPORTED'
                    ? 'text-accent-teal'
                    : standardRAG.evidenceSupport === 'PARTIAL'
                    ? 'text-accent-amber'
                    : 'text-accent-coral'
                }`}>
                  {standardRAG.evidenceSupport}
                </span>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-border/60">
                <span className="text-foreground-muted font-sans">Attribution Citations:</span>
                <span className="text-foreground-subtle">None (0.00% Coverage)</span>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-border/60">
                <span className="text-foreground-muted font-sans">LLM Generation Called:</span>
                <span className="text-accent-coral font-medium">Yes (100% of Queries)</span>
              </div>
            </div>
          </div>

          {/* Footer Metadata */}
          <div className="pt-4 border-t border-border/60 flex items-center justify-between text-xs font-mono text-foreground-subtle">
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5" />
              Latency: {standardRAG.latencyMs.toFixed(1)} ms
            </span>
            <span>Unverified Generation</span>
          </div>
        </div>

        {/* ClearRAG Column */}
        <div className="p-7 rounded-2xl bg-surface-100 border border-accent-teal/40 bg-accent-teal/5 flex flex-col justify-between">
          <div>
            {/* Header */}
            <div className="flex items-center justify-between mb-4 pb-4 border-b border-border/60">
              <div>
                <span className="text-xs font-mono uppercase text-accent-teal block mb-0.5">
                  Verified Architecture
                </span>
                <h3 className="text-base font-medium text-foreground">
                  ClearRAG
                </h3>
              </div>
              <span className={`text-xs font-mono px-2.5 py-1 rounded border font-medium ${
                isAbstain
                  ? 'bg-accent-teal/20 text-accent-teal border-accent-teal/40'
                  : clearRAG.decision === 'ANSWER_WITH_CAVEAT'
                  ? 'bg-accent-amber/20 text-accent-amber border-accent-amber/40'
                  : 'bg-accent-teal/20 text-accent-teal border-accent-teal/40'
              }`}>
                {clearRAG.decision}
              </span>
            </div>

            {/* Response Box */}
            <div className="p-5 rounded-xl bg-surface-200/70 border border-border/70 mb-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono uppercase text-accent-teal block">
                  {isAbstain ? 'Safe Abstention Decision' : 'Attributed Response'}
                </span>
                {!isAbstain && (
                  <span className="text-[10px] font-mono text-foreground-muted">
                    Click [1], [2] to inspect source chunk
                  </span>
                )}
              </div>
              <p className="text-sm font-sans text-foreground leading-relaxed">
                {renderAttributedAnswer(clearRAG.answer)}
              </p>
            </div>

            {/* Evidence & Grounding Diagnostics */}
            <div className="space-y-3 font-mono text-xs mb-6">
              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-border/60">
                <span className="text-foreground-muted font-sans">Verification Status:</span>
                <span className="text-accent-teal font-medium">
                  {clearRAG.verificationDetails.status}
                </span>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-border/60">
                <span className="text-foreground-muted font-sans">Attribution Precision:</span>
                <span className="text-accent-teal font-medium">
                  {(clearRAG.attributionPrecision * 100).toFixed(0)}% Grounded
                </span>
              </div>

              <div className="flex items-center justify-between p-3 rounded-lg bg-surface-200/50 border border-border/60">
                <span className="text-foreground-muted font-sans">LLM Compute Status:</span>
                {clearRAG.generationSkipped ? (
                  <span className="text-accent-teal font-semibold flex items-center gap-1">
                    <ZapOff className="w-3.5 h-3.5" />
                    GENERATION SKIPPED (0 Tokens)
                  </span>
                ) : (
                  <span className="text-foreground-muted">
                    Invoked with Verified Context
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Footer Metadata */}
          <div className="pt-4 border-t border-border/60 flex items-center justify-between text-xs font-mono text-accent-teal">
            <span className="flex items-center gap-1.5 text-foreground-subtle">
              <Clock className="w-3.5 h-3.5" />
              Latency: {clearRAG.latencyMs.toFixed(1)} ms
            </span>
            <span>Evidence-Gated Decision</span>
          </div>
        </div>
      </div>
    </div>
  );
};
