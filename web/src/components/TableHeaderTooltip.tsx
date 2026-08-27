import React, { useState } from 'react';
import { Info } from 'lucide-react';

interface TableHeaderTooltipProps {
  label: string;
  calculation: string;
  align?: 'left' | 'center' | 'right';
}

export const TableHeaderTooltip: React.FC<TableHeaderTooltipProps> = ({
  label,
  calculation,
  align = 'left',
}) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      className={`relative inline-flex items-center gap-1 cursor-help group select-none ${
        align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : 'justify-start'
      }`}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <span>{label}</span>
      <Info className="w-3.5 h-3.5 text-foreground-muted group-hover:text-accent-teal transition-colors shrink-0" />

      {isOpen && (
        <div
          className={`absolute z-[100] top-full mt-2 w-72 p-3.5 rounded-lg bg-surface-100 border border-accent-teal/40 shadow-2xl backdrop-blur-md text-left font-sans normal-case tracking-normal pointer-events-none animate-fade-in ${
            align === 'right'
              ? 'right-0'
              : align === 'center'
              ? 'left-1/2 -translate-x-1/2'
              : 'left-0'
          }`}
        >
          <div className="text-[10px] font-mono uppercase text-accent-teal tracking-wider mb-1 font-semibold">
            Calculation Rule
          </div>
          <p className="text-xs text-foreground font-normal leading-relaxed">
            {calculation}
          </p>
        </div>
      )}
    </div>
  );
};
