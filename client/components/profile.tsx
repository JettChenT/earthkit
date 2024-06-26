import { SignUpButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Button } from "./ui/button";
import { LogInIcon } from "lucide-react";
import { useEKGlobals } from "@/lib/globals";

export default function Profile() {
  let { sidebarExpanded } = useEKGlobals();
  return (
    <div className={sidebarExpanded ? "px-2 py-2" : ""}>
      <SignedIn>
        <div className={sidebarExpanded ? "" : "flex w-full justify-center"}>
          <UserButton
            appearance={{
              elements: {
                userButtonBox: {
                  flexDirection: "row-reverse",
                },
              },
            }}
            showName={sidebarExpanded}
            userProfileMode="modal"
          />
        </div>
      </SignedIn>
      <SignedOut>
        <SignUpButton>
          <Button variant="ghost" className="flex items-center gap-2">
            <LogInIcon className="size-4" />
            {sidebarExpanded && "Log In / Sign Up"}
          </Button>
        </SignUpButton>
      </SignedOut>
    </div>
  );
}
