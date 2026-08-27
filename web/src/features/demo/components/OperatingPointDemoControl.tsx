import React, { useState } from 'react';
import { Sliders, Shield, Zap, TrendingUp } from 'lucide-react';

interface OperatingPointSetting {
  id: string;
  name: string;
  threshold: number;
  coverage: string;
  answeredF1: string;
  unsupportedRate: string;
  computeSaved: string;
  description: string;
}

const OPERATING_POINTS: OperatingPointSetting[] = [
  {
    id: 'op-01',
    name: 'Setting 1 • Ultra-Safe',
    threshold: 0.90,
    coverage: '14.80%',
    answeredF1: '0.1920',
    unsupportedRate: '0.80%',
    computeSaved: '85.20%',
    description: 'Requires near-exact semantic match (θ=0.90). Virtually eliminates all hallucinations (<0.8%) at the expense of lower answer coverage.',
  },
  {
    id: 'op-04',
    name: 'Setting 4 • Default Calibrated',
    threshold: 0.75,
    coverage: '27.60%',
    answeredF1: '0.1685',
    unsupportedRate: '3.20%',
    computeSaved: '72.40%',
    description: 'The canonical research configuration (θ=0.75). Delivers a 91.4% relative reduction in hallucinations while answering high-confidence multi-hop queries.',
  },
  {
    id: 'op-10',
    name: 'Setting 10 • Max Quality',
    threshold: 0.45,
    coverage: '63.80%',
    answeredF1: '0.2014',
    unsupportedRate: '12.00%',
    computeSaved: '36.20%',
    description: 'Permissive verification threshold (θ=0.45). Recovers 36.84% of the raw Token F1 gap against Standard RAG while still cutting hallucinations by 67.6%.',
  },
];

export const OperatingPointDemoControl: React.FC = () => {
  const [selectedOp, setSelectedOp] = useState<string>('op-04');
  const currentOp = OPERATING_POINTS.find((o) => o.id === selectedOp) || OPERATING_POINTS[1];

  return (
    <div className="p-7 rounded-2xl bg-surface-100 border border-border mb-12">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6 pb-4 border-b border-border/60">
        <div>
          <span className="text-[10px] font-mono uppercase text-accent-teal tracking-wider block mb-0.5">
            Step 5 • Response Policy Tuning
          </span>
          <h3 className="text-base font-medium text-foreground">
            Configurable Verification Strictness (θ Threshold)
          </h3>
        </div>
        <span className="text-xs font-mono text-foreground-muted">
          Dynamic Coverage–Risk Tradeoff
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
        {OPERATING_POINTS.map((op) => {
          const isSelected = selectedOp === op.id;
          return (
            <button
              key={op.id}
              onClick={() => setSelectedOp(op.id)}
              className={`p-4 rounded-xl text-left border transition-all ${
                isSelected
                  ? 'bg-accent-teal/10 border-accent-teal/60 text-foreground'
                  : 'bg-surface-200/60 border-border text-foreground-muted hover:text-foreground hover:bg-surface-200'
              }`}
            >
              <div className="text-xs font-mono font-medium mb-1">{op.name}</div>
              <div className="text-[11px] font-mono text-accent-teal mb-2">
                Threshold θ = {op.threshold.toFixed(2)}
              </div>
              <div className="space-y-1 text-[10px] font-mono text-foreground-subtle">
                <div className="flex justify-between">
                  <span>Coverage:</span>
                  <span className="text-foreground">{op.coverage}</span>
                </div>
                <div className="flex justify-between">
                  <span>Hallucinations:</span>
                  <span className="text-foreground">{op.unsupportedRate}</span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="p-4 rounded-xl bg-surface-200/70 border border-border/70 text-xs font-sans text-foreground-muted leading-relaxed">
        <strong className="text-foreground font-mono text-[11px] block mb-1">
          {currentOp.name} (Threshold θ = {currentOp.threshold.toFixed(2)})
        </strong>
        {currentOp.description}
      </div>
    </div>
  );
};
