import React from 'react';
import { Layers, HelpCircle, Sparkles, AlertCircle, ArrowRight, Play } from 'lucide-react';
import { DEMO_SCENARIOS, ScenarioCategory, DemoQuestionInstance } from '@/data/demo_scenarios';

interface ScenarioSelectorProps {
  selectedCategory: string;
  onSelectCategory: (categoryId: string) => void;
  selectedQuestion: DemoQuestionInstance;
  onSelectQuestion: (question: DemoQuestionInstance) => void;
  customQuery: string;
  onChangeCustomQuery: (q: string) => void;
  onRunComparison: () => void;
  isRunning: boolean;
  hasRun: boolean;
}

export const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({
  selectedCategory,
  onSelectCategory,
  selectedQuestion,
  onSelectQuestion,
  customQuery,
  onChangeCustomQuery,
  onRunComparison,
  isRunning,
  hasRun,
}) => {
  const currentCategoryObj = DEMO_SCENARIOS.find((c) => c.id === selectedCategory) || DEMO_SCENARIOS[0];

  return (
    <div className="mb-12 space-y-6">
      {/* 5 Scenario Category Tabs */}
      <div>
        <span className="text-[11px] font-mono uppercase tracking-widest text-accent-teal block mb-3">
          Step 1 • Select Benchmark Evidence Condition
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5">
          {DEMO_SCENARIOS.map((cat) => {
            const isSelected = selectedCategory === cat.id;
            return (
              <button
                key={cat.id}
                onClick={() => {
                  onSelectCategory(cat.id);
                  if (cat.questions.length > 0) {
                    onSelectQuestion(cat.questions[0]);
                  }
                }}
                className={`p-3.5 rounded-xl text-left border transition-all ${
                  isSelected
                    ? 'bg-accent-teal/10 border-accent-teal/60 text-foreground shadow-sm'
                    : 'bg-surface-100 border-border text-foreground-muted hover:border-border/80 hover:text-foreground'
                }`}
              >
                <div className="text-xs font-mono font-medium mb-1 line-clamp-1">
                  {cat.label}
                </div>
                <div className="text-[10px] font-sans text-foreground-muted/80 line-clamp-2">
                  {cat.description}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Scenario Explanation Card */}
      <div className="p-4 rounded-xl bg-surface-100 border border-border/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-sans">
        <div>
          <span className="text-[10px] font-mono uppercase text-accent-teal block mb-0.5">
            {currentCategoryObj.badge}
          </span>
          <span className="text-foreground-muted">
            <strong className="text-foreground">Expected Research Outcome:</strong> {currentCategoryObj.expectedBehavior}
          </span>
        </div>
      </div>

      {/* Curated Question Pickers vs Custom Query */}
      <div className="space-y-3">
        <span className="text-[11px] font-mono uppercase tracking-widest text-foreground-muted block">
          Step 2 • Select a Question
        </span>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {currentCategoryObj.questions.map((q) => {
            const isSelected = selectedQuestion.id === q.id && !customQuery;
            return (
              <button
                key={q.id}
                onClick={() => {
                  onChangeCustomQuery('');
                  onSelectQuestion(q);
                }}
                className={`p-4 rounded-xl text-left border transition-all ${
                  isSelected
                    ? 'bg-surface-200 border-accent-teal/60 text-foreground'
                    : 'bg-surface-100 border-border text-foreground-muted hover:text-foreground hover:bg-surface-100/80'
                }`}
              >
                <span className="text-[10px] font-mono uppercase text-foreground-subtle block mb-1">
                  Authentic HotpotQA Query
                </span>
                <p className="text-xs font-sans text-foreground font-medium mb-2 leading-relaxed">
                  "{q.question}"
                </p>
                <span className="text-[11px] font-mono text-accent-teal">
                  Gold Reference: {q.goldAnswer}
                </span>
              </button>
            );
          })}
        </div>

        {/* Custom Question Bar */}
        <div className="pt-2">
          <div className="p-2 rounded-xl bg-surface-100 border border-border flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
            <input
              type="text"
              placeholder="Or enter your own custom question to test both response policies..."
              value={customQuery}
              onChange={(e) => onChangeCustomQuery(e.target.value)}
              className="flex-1 px-4 py-2.5 text-xs rounded-lg bg-surface-200/80 border border-border text-foreground placeholder:text-foreground-muted/60 focus:outline-none focus:border-accent-teal font-sans"
            />
            <button
              onClick={onRunComparison}
              disabled={isRunning}
              className="px-6 py-2.5 rounded-lg bg-accent-teal text-background font-mono text-xs font-semibold hover:bg-accent-teal/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2 shrink-0 shadow-md"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>{isRunning ? 'Executing Pipeline...' : hasRun ? 'Re-run Comparison' : 'Run Comparison'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
