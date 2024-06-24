"use client";
import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { useEffect, useMemo, useState } from "react";
import { useUser } from "@clerk/nextjs";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline:
          "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-6 rounded-md px-3",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  regEvent?: string;
  eventGuard?: (event: Event) => boolean;
  requireLogin?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      asChild = false,
      regEvent,
      eventGuard,
      onClick,
      onMouseDown,
      disabled,
      requireLogin,
      children,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : "button";
    useEffect(() => {
      if (!regEvent) return;

      const handler = (event: Event) => {
        if (disabled || (eventGuard && !eventGuard(event))) return;
        onClick?.(new MouseEvent("click", event) as any);
        onMouseDown?.(new MouseEvent("mousedown", event) as any);
      };

      document.addEventListener(regEvent, handler);
      return () => {
        document.removeEventListener(regEvent, handler);
      };
    }, [regEvent, eventGuard, disabled, onClick, onMouseDown]);

    const user = useUser();
    const unauthorized = useMemo(() => {
      if (!requireLogin) return false;
      if (user) return false;
      return true;
    }, [user, requireLogin]);
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        onClick={onClick}
        onMouseDown={onMouseDown}
        disabled={disabled || unauthorized}
        {...props}
      >
        {unauthorized ? (
          <>
            {children}
            <span className="text-secondary-foreground">(Requires Login)</span>
          </>
        ) : (
          children
        )}
      </Comp>
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
