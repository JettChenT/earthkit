import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Navbar() {
  return (
    <header className="flex justify-between w-full items-center drop-shadow-sm bg-gray-50 border-gray-100 text-gray-700 rounded-xl py-2 px-4">
      <Link href="/" className="font-bold text-3xl font-mono drop-shadow-sm">
        <span className="text-blue-700">E</span>
        <span className="text-green-700">K</span>
      </Link>
      <div className="space-x-4 flex items-center">
        <Link
          href="https://agent.earthkit.app"
          className="underline text-sm variant-ghost"
        >
          Agent
        </Link>
        <Link
          href="https://discord.gg/X3YRuwZBNn"
          className="underline text-sm variant-ghost"
        >
          Discord
        </Link>
        <Link
          href="https://t.me/+FUm4YxEAfX8yZTA9"
          className="underline text-sm variant-ghost"
        >
          Telegram
        </Link>
        <Link
          href="https://github.com/JettChenT/earthkit"
          className="underline text-sm variant-ghost"
        >
          GitHub
        </Link>
        <Button asChild size="sm">
          <Link href="/sift" className="font-bold font-mono">
            App
          </Link>
        </Button>
      </div>
    </header>
  );
}
