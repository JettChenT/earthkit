"use client";
import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";

import { sideBarData } from "./sidebar";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export type CommandItemData = {
  type: "CommandItemData";
  icon?: React.ReactNode;
  disabled?: boolean;
  display: string;
  event?: string;
  action?: () => unknown;
};

export type CommandGroupData = {
  type: "CommandGroupData";
  heading: string;
  children: CommandItemData[];
};

export type CommandsData = (CommandItemData | CommandGroupData)[];

export type Listener = {
  event: string;
  handler: () => unknown;
};

export function useListeners(listeners: Listener[]) {
  useEffect(() => {
    listeners.forEach((listener) => {
      document.addEventListener(listener.event, listener.handler);
    });
    return () => {
      listeners.forEach((listener) => {
        document.removeEventListener(listener.event, listener.handler);
      });
    };
  }, [listeners]);
}

export function CommandBar({ commands }: { commands: CommandsData }) {
  const router = useRouter();
  let [open, setOpen] = useState(false);
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };
    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);
  const runCommand = (fnc: () => unknown) => {
    setOpen(false);
    fnc();
  };

  const renderItem = (item: CommandItemData, idx: number) => {
    if (item.disabled) return null;
    return (
      <CommandItem
        key={idx}
        className="flex items-center gap-2 justify-start"
        onSelect={() => {
          runCommand(() => {
            if (item.event) {
              document.dispatchEvent(new Event(item.event));
            }
            item.action?.();
          });
        }}
      >
        {item.icon}
        {item.display}
      </CommandItem>
    );
  };

  const renderGroup = (group: CommandGroupData, idx: number) => {
    return (
      <CommandGroup key={idx} heading={group.heading}>
        {group.children.map((item, idx) => renderItem(item, idx))}
      </CommandGroup>
    );
  };

  const renderCommands = (commands: CommandsData) => {
    return commands.map((command, idx) => {
      if (command.type === "CommandItemData") {
        return renderItem(command, idx);
      }
      return renderGroup(command, idx);
    });
  };

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <Command className="rounded-lg border shadow-md">
        <CommandInput placeholder="Type a command or search..." />
        <CommandList>
          <CommandEmpty>No results found.</CommandEmpty>
          {renderCommands(commands)}
          <CommandGroup heading="Navigation">
            {sideBarData.map((item, idx) => (
              <CommandItem
                key={idx}
                onSelect={() => {
                  runCommand(() => {
                    router.push(item.tool);
                  });
                }}
                className="flex items-center gap-2 justify-start"
              >
                {item.icon}
                Go To {item.display}
              </CommandItem>
            ))}
          </CommandGroup>
        </CommandList>
      </Command>
    </CommandDialog>
  );
}

const DefaultPathList = ["/streetview", "/satellite", "/geoclip"];

export function DefaultKBar() {
  const pathname = usePathname();
  if (DefaultPathList.includes(pathname)) {
    return <CommandBar commands={[]} />;
  }
  return null;
}
