import { cn } from "@/lib/utils";

type StatusLevel = "High" | "Medium" | "Low" | "Unknown";

interface StatusBadgeProps {
  level: StatusLevel;
  className?: string;
  withDot?: boolean;
}

export function StatusBadge({ level, className, withDot = true }: StatusBadgeProps) {
  const styles = {
    High: "bg-destructive/10 text-destructive border-destructive/20",
    Medium: "bg-warning/10 text-warning border-warning/20",
    Low: "bg-success/10 text-success border-success/20",
    Unknown: "bg-muted text-muted-foreground border-border",
  };

  const dotStyles = {
    High: "bg-destructive",
    Medium: "bg-warning",
    Low: "bg-success",
    Unknown: "bg-muted-foreground",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border",
        styles[level] || styles.Unknown,
        className
      )}
    >
      {withDot && (
        <span className="relative flex h-2 w-2">
          {level !== "Unknown" && (
            <span
              className={cn(
                "animate-ping absolute inline-flex h-full w-full rounded-full opacity-40",
                dotStyles[level]
              )}
            ></span>
          )}
          <span className={cn("relative inline-flex rounded-full h-2 w-2", dotStyles[level])}></span>
        </span>
      )}
      {level}
    </span>
  );
}
