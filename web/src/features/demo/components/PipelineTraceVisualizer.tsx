import React from 'react';
import { Check, ArrowRight, ShieldCheck, Cpu, ZapOff, Sparkles } from 'lucide-react';
import { DemoQuestionInstance } from '@/data/demo_scenarios';

interface PipelineTraceVisualizerProps {
  question: DemoQuestionInstance;
  isRunning: boolean;
}

export const PipelineTraceVisualizer: React.FC<PipelineTraceVisualizerProps> = ({ question, isRunning }) => {
  const isAbstain = question.clearRAG.decision.includes('ABSTAIN');

  return (
    <div className="p-6 rounded-2xl bg-surface-100 border border-border mb-12">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-border/60">
        <div>
          <span className="text-[10px] font-mono uppercase text-accent-teal tracking-wider block mb-0.5">
            Architecture Trace
          </span>
          <h3 className="text-sm font-medium text-foreground">
            Pipeline Execution Comparison
          </h3>
        </div>
        <span className="text-xs font-mono text-foreground-muted">
          Same Question → Two Response Policies
        </span>
      </div>

      <div className="space-y-6 text-xs font-mono">
        {/* Standard RAG Line */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-foreground-muted uppercase text-[11px]">
              Standard RAG Pipeline:
            </span>
            <span className="text-[11px] text-accent-coral">
              Unconditional Always-Answer Policy
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
            <div className="p-3 rounded-lg bg-surface-200/80 border border-border flex items-center justify-between">
              <span className="text-foreground">1. Dense FAISS (k=5)</span>
              <Check className="w-3.5 h-3.5 text-foreground-muted" />
            </div>
            <div className="p-3 rounded-lg bg-surface-200/80 border border-border flex items-center justify-between">
              <span className="text-foreground">2. Prompt Assembly</span>
              <Check className="w-3.5 h-3.5 text-foreground-muted" />
            </div>
            <div className="p-3 rounded-lg bg-accent-coral/10 border border-accent-coral/40 text-accent-coral flex items-center justify-between">
              <span className="font-semibold">3. LLM Generation (Always)</span>
              <Cpu className="w-3.5 h-3.5 text-accent-coral" />
            </div>
          </div>
        </div>

        {/* ClearRAG Line */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-accent-teal uppercase text-[11px]">
              ClearRAG Pipeline:
            </span>
            <span className="text-[11px] text-accent-teal">
              Evidence-Gated Verification Control
            </span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
            <div className="p-3 rounded-lg bg-surface-200/80 border border-border flex items-center justify-between">
              <span className="text-foreground">1. Hybrid (k=10)</span>
              <Check className="w-3.5 h-3.5 text-accent-teal" />
            </div>
            <div className="p-3 rounded-lg bg-surface-200/80 border border-border flex items-center justify-between">
              <span className="text-foreground">2. Claim Verifier</span>
              <Check className="w-3.5 h-3.5 text-accent-teal" />
            </div>
            <div className="p-3 rounded-lg bg-surface-200/80 border border-border flex items-center justify-between">
              <span className="text-foreground">3. Decision Policy</span>
              <Check className="w-3.5 h-3.5 text-accent-teal" />
            </div>

            {/* Generation Status (Highlighted or Skipped) */}
            {isAbstain ? (
              <div className="p-3 rounded-lg bg-accent-teal/20 border border-accent-teal text-accent-teal flex items-center justify-between sm:col-span-2">
                <div className="flex items-center gap-2">
                  <ZapOff className="w-4 h-4 text-accent-teal" />
                  <div>
                    <strong className="block text-[11px]">4. GENERATION SKIPPED</strong>
                    <span className="text-[10px] text-foreground-muted font-sans">
                      GPU compute saved: evidence insufficient
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="p-3 rounded-lg bg-surface-200/80 border border-border flex items-center justify-between">
                  <span className="text-foreground">4. Grounded Gen</span>
                  <Cpu className="w-3.5 h-3.5 text-accent-teal" />
                </div>
                <div className="p-3 rounded-lg bg-accent-teal/10 border border-accent-teal/40 text-accent-teal flex items-center justify-between">
                  <span className="font-semibold">5. Attribution</span>
                  <Check className="w-3.5 h-3.5 text-accent-teal" />
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
