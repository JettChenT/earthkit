"use client";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
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
import { Textarea } from "@/components/ui/textarea";
import { usePathname } from "next/navigation";
import { ReactNode, useMemo, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";
import Kbd, { MetaKey } from "./keyboard";
import { sideBarData } from "./sidebar";
import ky from "ky";
import { toast } from "sonner";

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
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  useHotkeys("Meta + /", () => {
    setShortcutOpen((prev) => !prev);
  });
  return (
    <div className="absolute right-4 bottom-4 z-40">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="bg-slate-900 rounded-full size-6 text-white border border-slate-600">
            ?
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-[200px]" align="end">
          <DropdownMenuItem onClick={() => setShortcutOpen(true)}>
            Keyboard Shortcuts
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setFeedbackOpen(true)}>
            Send Feedback
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => {
              window.open("https://cal.com/jettchent/30min");
            }}
          >
            Schedule a Call
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <KeyboardCheetSheet open={shortcutOpen} onOpenChange={setShortcutOpen} />
      <FeedbackDialog open={feedbackOpen} onOpenChange={setFeedbackOpen} />
    </div>
  );
}

type Section = {
  title: string;
  items: KeyboardShortcutItem[];
};

function KeyboardCheetSheet({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
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
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-xl">Keyboard Shortcuts</DialogTitle>
        </DialogHeader>
        <DialogDescription>
          <div>
            {sections.map((section) => {
              return (
                <div key={section.title}>
                  <h3 className="text-md text-black font-bold">
                    {section.title}
                  </h3>
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
        </DialogDescription>
      </DialogContent>
    </Dialog>
  );
}

function FeedbackDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  let [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);
  const onSubmit = async () => {
    setLoading(true);
    await ky.post("/api/feedback", { json: { feedback } });
    toast.success("Feedback submitted");
    setLoading(false);
    setFeedback("");
    onOpenChange(false);
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Feedback</DialogTitle>
        </DialogHeader>
        <DialogDescription>
          EarthKit is still in active development and have lots of areas to
          improve! Send anonymized feedback to help us make it better.
          <Textarea
            className="mt-3"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="How can EarthKit improve? Notice any annoying bugs? "
          />
        </DialogDescription>
        <DialogFooter>
          <Button onClick={onSubmit} disabled={loading}>
            {loading ? "Submitting..." : "Submit"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
