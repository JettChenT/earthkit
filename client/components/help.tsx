"use client";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  NormalTable,
  TableBody,
  TableCell,
  TableRow,
} from "@/components/ui/table";
import { usePathname } from "next/navigation";
import { ReactNode, useMemo, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import Kbd, { MetaKey } from "./keyboard";
import { sideBarData } from "./sidebar";

type KeyboardShortcutItem = {
  key: ReactNode;
  description: ReactNode;
};

const shortCuts: Record<string, KeyboardShortcutItem[]> = {
  navigation: [
    {
      key: (
        <>
          <MetaKey /> + <Kbd>K</Kbd>
        </>
      ),
      description: "Command Bar",
    },
    {
      key: (
        <>
          <MetaKey /> + <Kbd>/</Kbd>
        </>
      ),
      description: "Show Keyboard Shortcuts",
    },
  ],
  sift: [
    {
      key: (
        <>
          <Kbd>↑</Kbd> or <Kbd>k</Kbd>
        </>
      ),
      description: "Move Selection Up",
    },
    {
      key: (
        <>
          <Kbd>↓</Kbd> or <Kbd>j</Kbd>
        </>
      ),
      description: "Move Selection Down",
    },
    {
      key: (
        <>
          <Kbd>h</Kbd> or <Kbd>n</Kbd> or <Kbd>1</Kbd>
        </>
      ),
      description: "Label: Not Match",
    },
    {
      key: (
        <>
          <Kbd>p</Kbd> or <Kbd>2</Kbd>
        </>
      ),
      description: "Label: Keep",
    },
    {
      key: (
        <>
          <Kbd>l</Kbd> or <Kbd>m</Kbd> or <Kbd>3</Kbd>
        </>
      ),
      description: "Label: Match",
    },
  ],
  osm: [
    {
      key: <Kbd>Enter</Kbd>,
      description: "Submit Query",
    },
    {
      key: (
        <>
          <Kbd>Ctrl</Kbd> + <Kbd>Space</Kbd>
        </>
      ),
      description: "Toggle OSM Completion",
    },
  ],
  geoclip: [
    {
      key: (
        <>
          <MetaKey /> + <Kbd>C</Kbd>
        </>
      ),
      description: "Copy Cursor Coordinates",
    },
  ],
};

export default function Help() {
  const [shortcutOpen, setShortcutOpen] = useState(false);
  useHotkeys("Meta + /", () => {
    setShortcutOpen((prev) => !prev);
  });
  return (
    <div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="bg-slate-900 rounded-full size-6 text-white">
            ?
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-[200px]" align="end">
          <DropdownMenuItem onClick={() => setShortcutOpen(true)}>
            Keyboard Shortcuts
            <DropdownMenuShortcut>?</DropdownMenuShortcut>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <Dialog open={shortcutOpen} onOpenChange={setShortcutOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-xl">Keyboard Shortcuts</DialogTitle>
          </DialogHeader>
          <DialogDescription>
            <KeyboardCheetSheet />
          </DialogDescription>
        </DialogContent>
      </Dialog>
    </div>
  );
}

type Section = {
  title: string;
  items: KeyboardShortcutItem[];
};

function KeyboardCheetSheet() {
  const pathname = usePathname();
  const sections = useMemo(() => {
    let res = [
      {
        title: "Navigation",
        items: shortCuts.navigation,
      },
    ];
    const toolName = pathname.slice(1);
    const sec = shortCuts[toolName];
    if (sec) {
      res.push({
        title:
          sideBarData.find((item) => item.tool === toolName)?.display || "",
        items: sec,
      });
    }
    return res;
  }, [pathname]);
  console.log(sections);
  return (
    <div>
      {sections.map((section) => {
        return (
          <div key={section.title}>
            <h3 className="text-md text-black font-bold">{section.title}</h3>
            <hr className="my-2 -mb-1" />
            <NormalTable>
              <TableBody>
                {section.items.map((item, idx) => {
                  return (
                    <TableRow key={idx}>
                      <TableCell className="w-40">{item.key}</TableCell>
                      <TableCell>{item.description}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </NormalTable>
          </div>
        );
      })}
    </div>
  );
}
