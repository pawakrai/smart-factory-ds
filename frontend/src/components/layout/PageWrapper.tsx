interface PageWrapperProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}

import React from "react";

export default function PageWrapper({ title, subtitle, children, actions }: PageWrapperProps) {
  return (
    <div className="max-w-screen-2xl mx-auto px-6 py-6">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-[var(--text-primary)] leading-tight">{title}</h1>
          {subtitle && <p className="text-sm text-zinc-400 mt-0.5">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
      {children}
    </div>
  );
}
