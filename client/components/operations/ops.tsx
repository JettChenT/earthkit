"use client";

import { twMerge } from "tailwind-merge";

export default function OperationContainer({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={twMerge(
        "w-72 p-3 bg-opacity-80 absolute right-3 mt-3 bg-white rounded-md",
        className
      )}
    >
      {children}
    </div>
  );
}
