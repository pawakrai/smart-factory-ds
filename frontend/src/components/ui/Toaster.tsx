import * as RadixToast from "@radix-ui/react-toast";
import { CheckCircle2, AlertCircle, Info, X } from "lucide-react";
import { useToastStore } from "@/store/toastStore";
import type { ToastType } from "@/store/toastStore";

const icons: Record<ToastType, React.ReactNode> = {
  success: <CheckCircle2 size={15} className="text-green-400 shrink-0" />,
  error: <AlertCircle size={15} className="text-brand-red shrink-0" />,
  info: <Info size={15} className="text-blue-400 shrink-0" />,
};

const borders: Record<ToastType, string> = {
  success: "border-green-800/60",
  error: "border-red-900/60",
  info: "border-blue-800/60",
};

import React from "react";

export default function Toaster() {
  const { toasts, removeToast } = useToastStore();

  return (
    <RadixToast.Provider swipeDirection="right" duration={4000}>
      {toasts.map((toast) => (
        <RadixToast.Root
          key={toast.id}
          open={true}
          onOpenChange={(open) => !open && removeToast(toast.id)}
          className={`
            group flex items-start gap-3 bg-bg-elevated border ${borders[toast.type]}
            rounded-xl px-4 py-3 shadow-2xl
            data-[state=open]:animate-in data-[state=open]:slide-in-from-right-full
            data-[state=closed]:animate-out data-[state=closed]:slide-out-to-right-full
            transition-all duration-200
          `}
        >
          {icons[toast.type]}
          <div className="flex-1 min-w-0">
            <RadixToast.Title className="text-sm font-medium text-[var(--text-primary)] leading-tight">
              {toast.title}
            </RadixToast.Title>
            {toast.description && (
              <RadixToast.Description className="text-xs text-zinc-400 mt-0.5">
                {toast.description}
              </RadixToast.Description>
            )}
          </div>
          <RadixToast.Close
            onClick={() => removeToast(toast.id)}
            className="text-zinc-600 hover:text-[var(--text-secondary)] transition-colors mt-0.5"
          >
            <X size={13} />
          </RadixToast.Close>
        </RadixToast.Root>
      ))}

      <RadixToast.Viewport className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 w-80 max-w-[calc(100vw-2rem)]" />
    </RadixToast.Provider>
  );
}
