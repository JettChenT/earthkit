import { SignInButton, SignedIn, SignedOut, UserButton } from "@clerk/nextjs";

export default function Profile() {
  return (
    <div className="px-7 py-2">
      <SignedIn>
        <UserButton showName={true} userProfileMode="modal" />
      </SignedIn>
      <SignedOut>
        <SignInButton />
      </SignedOut>
    </div>
  );
}
