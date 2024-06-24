import { SignUpButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";
import { Button } from "./ui/button";
import { LogInIcon } from "lucide-react";

export default function Profile() {
  return (
    <div className="px-6 py-2">
      <SignedIn>
        <UserButton
          appearance={{
            elements: {
              userButtonBox: {
                flexDirection: "row-reverse",
              },
            },
          }}
          showName={true}
          userProfileMode="modal"
        />
      </SignedIn>
      <SignedOut>
        <SignUpButton>
          <Button variant="ghost" className="flex items-center">
            <LogInIcon className="size-4 mr-2" />
            Log In / Sign Up
          </Button>
        </SignUpButton>
      </SignedOut>
    </div>
  );
}
