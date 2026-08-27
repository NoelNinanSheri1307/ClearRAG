import React, { useState } from 'react';
import { ZoomIn, Info, BarChart2, HelpCircle } from 'lucide-react';
import { CANONICAL_RESEARCH_DATA } from '@/data/canonical_research_data';
import { PlotModal } from '@/components/PlotModal';

export const ParetoGallerySection: React.FC = () => {
  const { publication_plots } = CANONICAL_RESEARCH_DATA;
  const [selectedPlot, setSelectedPlot] = useState<typeof publication_plots[0] | null>(null);

  return (
    <section id="pareto" className="py-24 px-6 border-t border-border/60 bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="mb-10 max-w-3xl">
          <div className="text-xs font-mono uppercase tracking-widest text-accent-teal mb-3">
            14 / Research Visualizations
          </div>
          <h2 className="text-3xl sm:text-5xl font-serif text-foreground font-normal tracking-tight mb-4">
            Research Charts & Tradeoff Curves
          </h2>
          <p className="text-base text-foreground-muted font-sans font-light leading-relaxed mb-6">
            Empirical evaluation charts illustrating the tradeoff between factual safety, safe abstention, and answer coverage across benchmark conditions.
          </p>

          <div className="p-4 rounded-xl bg-surface-100 border border-border flex items-start gap-3 text-xs font-sans text-foreground-muted">
            <HelpCircle className="w-4 h-4 text-accent-teal shrink-0 mt-0.5" />
            <span>
              <strong className="text-foreground">What do these charts show?</strong> They visualize the balance between safety (abstaining on missing facts) and volume (answering questions), showing how tuning the verifier threshold lets you choose your preferred safety-vs-coverage balance.
            </span>
          </div>
        </div>

        {/* Featured Exhibit Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {publication_plots.slice(0, 9).map((plot) => (
            <div
              key={plot.title}
              onClick={() => setSelectedPlot(plot)}
              className="p-5 rounded-2xl bg-surface-100 border border-border hover:border-accent-teal/50 transition-all duration-300 cursor-pointer group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-surface-200 text-foreground-muted border border-border">
                    {plot.category}
                  </span>
                  <ZoomIn className="w-4 h-4 text-foreground-subtle group-hover:text-accent-teal transition-colors" />
                </div>

                <div className="h-44 rounded-lg bg-surface-200/80 border border-border/60 overflow-hidden mb-4 flex items-center justify-center p-2">
                  <img
                    src={`/plots/${plot.filename}`}
                    alt={plot.title}
                    className="max-h-full max-w-full object-contain rounded group-hover:scale-105 transition-transform duration-300"
                    loading="lazy"
                  />
                </div>

                <h3 className="text-sm font-medium text-foreground mb-1 group-hover:text-accent-teal transition-colors">
                  {plot.title}
                </h3>
                <p className="text-xs text-foreground-muted font-sans line-clamp-2">
                  {plot.caption}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-border/60 text-[10px] font-mono text-accent-teal group-hover:underline">
                View Full Chart & Details
              </div>
            </div>
          ))}
        </div>

        {/* Modal viewer */}
        <PlotModal
          isOpen={!!selectedPlot}
          onClose={() => setSelectedPlot(null)}
          plot={selectedPlot}
        />
      </div>
    </section>
  );
};
