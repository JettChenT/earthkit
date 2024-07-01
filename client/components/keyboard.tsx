import { cn } from "@/lib/utils";

export default function Kbd({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-block shadow-sm bg-gray-100 text-gray-700 text-xs font-mono font-semibold px-2 py-1 rounded border",
        className
      )}
    >
      {children}
    </span>
  );
}

export function MetaKey({
  noWrap,
  className,
}: {
  noWrap?: boolean;
  className?: string;
}) {
  const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
  const key = isMac ? "⌘" : "Ctrl";
  return noWrap ? <>{key}</> : <Kbd className={className}>{key}</Kbd>;
}

export function KbdContainer({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-2">{children}</div>;
}
