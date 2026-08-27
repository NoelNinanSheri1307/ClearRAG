import React, { useState } from 'react';
import { Search, BookOpen, Filter, ArrowUp, ArrowDown } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA, MetricDefinitionItem } from '@/data/canonical_research_data';

export const MetricDictionarySection: React.FC = () => {
  const { metric_dictionary } = CANONICAL_RESEARCH_DATA;
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', 'Decision & Safety', 'Generation Quality', 'Retrieval', 'Verification', 'Attribution', 'Efficiency'];

  const filteredMetrics = metric_dictionary.filter((m) => {
    const matchesSearch = m.name.toLowerCase().includes(searchTerm.toLowerCase()) || m.plain_english.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || m.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <section id="dictionary" className="py-24 px-6 border-t border-border/60 bg-surface-300/40">
      <div className="max-w-5xl mx-auto">
        <div className="mb-14">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            17 / Metric Dictionary
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Metric Reference
          </h2>

        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-stretch sm:items-center mb-8">
          {/* Category Tabs */}
          <div className="flex flex-wrap gap-2">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`text-xs font-mono px-3 py-1.5 rounded-lg border transition-colors ${selectedCategory === cat
                  ? 'bg-accent-teal/15 text-accent-teal border-accent-teal/30 font-medium'
                  : 'bg-surface-100 text-foreground-muted border-border hover:text-foreground'
                  }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Search Input */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-foreground-muted" />
            <input
              type="text"
              placeholder="Search metrics..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full sm:w-64 pl-9 pr-4 py-1.5 text-xs rounded-lg bg-surface-100 border border-border text-foreground placeholder:text-foreground-muted/60 focus:outline-none focus:border-accent-teal font-sans"
            />
          </div>
        </div>

        {/* Metric Cards Grid */}
        <div className="space-y-4">
          {filteredMetrics.map((metric) => (
            <div
              key={metric.id}
              className="p-6 rounded-2xl bg-surface-100 border border-border hover:border-border-highlight transition-colors"
            >
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-3">
                <div>
                  <div className="flex items-center gap-2.5 mb-1.5">
                    <h3 className="text-base font-medium text-foreground">
                      {metric.name}
                    </h3>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-200 text-foreground-muted border border-border">
                      {metric.category}
                    </span>
                    <span className="text-[10px] font-mono text-foreground-subtle flex items-center gap-1">
                      {metric.higher_is_better ? (
                        <>
                          <ArrowUp className="w-3 h-3 text-accent-teal" /> Higher is better
                        </>
                      ) : (
                        <>
                          <ArrowDown className="w-3 h-3 text-accent-coral" /> Lower is better
                        </>
                      )}
                    </span>
                  </div>
                  <p className="text-xs text-foreground-muted font-sans leading-relaxed">
                    {metric.plain_english}
                  </p>
                </div>

                {/* Score Comparison Badge */}
                <div className="flex items-center gap-3 shrink-0 font-mono text-xs border-t md:border-t-0 md:border-l border-border/60 pt-3 md:pt-0 md:pl-4">
                  <div className="text-right">
                    <span className="text-[10px] text-foreground-subtle block">Standard RAG</span>
                    <span className="text-foreground-muted">{metric.standard_rag_value}</span>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] text-accent-teal block">ClearRAG</span>
                    <span className="text-accent-teal font-medium">{metric.clearrag_default_value}</span>
                  </div>
                </div>
              </div>

              {/* Formula & Interpretation */}
              <div className="mt-4 pt-3 border-t border-border/60 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                {metric.formula && (
                  <div className="p-2.5 rounded bg-surface-200/80 border border-border/60 text-foreground overflow-x-auto text-[11px]">
                    <span className="text-foreground-subtle block text-[10px] uppercase mb-0.5">Formula</span>
                    {metric.formula}
                  </div>
                )}
                <div className="p-2.5 rounded bg-surface-200/80 border border-border/60 text-foreground-muted font-sans text-[11px] leading-relaxed">
                  <strong className="text-foreground font-mono uppercase text-[10px] block mb-0.5">Interpretation</strong>
                  {metric.interpretation}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
