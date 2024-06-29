export default function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-block shadow-sm bg-gray-100 text-gray-700 text-xs font-mono font-semibold px-2 py-1 rounded">
      {children}
    </span>
  );
}

export function MetaKey() {
  const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0;
  return <Kbd>{isMac ? "⌘" : "Ctrl"}</Kbd>;
}

export function KbdContainer({ children }: { children: React.ReactNode }) {
  return <div className="flex items-center gap-2">{children}</div>;
}
