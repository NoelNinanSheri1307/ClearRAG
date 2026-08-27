import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="py-8 px-6 border-t border-border/80 bg-surface-300">
      <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-foreground-subtle">
        <span>ClearRAG</span>
        <span className="text-foreground font-medium">Made by Noel Ninan Sheri</span>
        <span>Evidence-Aware Selective Generation</span>
      </div>
    </footer>
  );
};
