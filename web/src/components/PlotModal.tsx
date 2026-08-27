import React from 'react';
import { X, ZoomIn, Info } from 'lucide-react';

interface PlotModalProps {
  isOpen: boolean;
  onClose: () => void;
  plot: {
    filename: string;
    title: string;
    caption: string;
    category: string;
  } | null;
}

export const PlotModal: React.FC<PlotModalProps> = ({ isOpen, onClose, plot }) => {
  if (!isOpen || !plot) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm transition-opacity animate-fade-in"
      onClick={onClose}
    >
      <div 
        className="relative max-w-5xl w-full bg-surface-100 border border-border rounded-xl shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-surface-200/50">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono px-2.5 py-0.5 rounded bg-accent-teal/10 text-accent-teal border border-accent-teal/20">
              {plot.category}
            </span>
            <h3 className="text-lg font-medium text-foreground">{plot.title}</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-foreground-muted hover:text-foreground hover:bg-surface-50 rounded-lg transition-colors"
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Image Container */}
        <div className="p-6 bg-surface-300 flex items-center justify-center min-h-[400px] max-h-[70vh] overflow-auto">
          <img
            src={`/plots/${plot.filename}`}
            alt={plot.title}
            className="max-w-full max-h-[65vh] object-contain rounded-lg shadow-lg border border-border/50"
          />
        </div>

        {/* Footer / Caption */}
        <div className="p-6 bg-surface-100 border-t border-border flex items-start gap-3">
          <Info className="w-5 h-5 text-accent-teal shrink-0 mt-0.5" />
          <p className="text-sm text-foreground-muted leading-relaxed font-sans">
            {plot.caption}
          </p>
        </div>
      </div>
    </div>
  );
};
