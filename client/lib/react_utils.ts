"use client";
import { createRoot } from "react-dom/client";

export function renderReactNode(
  node: React.ReactNode,
  parent: string = "div",
  className?: string
): HTMLElement {
  const div = document.createElement(parent);
  if (className) {
    div.className = className;
  }
  const root = createRoot(div);
  root.render(node);
  return div;
}
