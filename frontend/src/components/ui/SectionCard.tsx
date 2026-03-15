import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface SectionCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  action?: ReactNode;
  delay?: number;
}

export function SectionCard({ children, className, title, action, delay = 0 }: SectionCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay, ease: "easeOut" }}
      className={cn(
        "bg-card rounded-2xl border border-border/60 organic-shadow overflow-hidden flex flex-col",
        className
      )}
    >
      {(title || action) && (
        <div className="px-6 py-5 border-b border-border/40 flex items-center justify-between bg-card/50">
          {title && <h3 className="text-lg font-display font-medium text-foreground">{title}</h3>}
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-6 flex-1">
        {children}
      </div>
    </motion.div>
  );
}
