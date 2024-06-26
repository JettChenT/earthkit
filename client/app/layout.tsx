import type { Metadata } from "next";
import { Inter as FontSans } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { Toaster } from "@/components/ui/sonner";
import Sidebar from "@/components/sidebar";
import { DefaultKBar } from "@/components/kbar";
import { Analytics } from "@vercel/analytics/react";
import {
  ClerkProvider,
  SignInButton,
  SignedIn,
  SignedOut,
  UserButton,
} from "@clerk/nextjs";

export const metadata: Metadata = {
  title: "EarthKit",
  description: "EarthKit is a tool for locating images on the Earth.",
};

const fontSans = FontSans({
  subsets: ["latin"],
  variable: "--font-sans",
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <ClerkProvider
      appearance={{
        elements: {
          socialButtons: {
            flexDirection: "column",
          },
        },
      }}
    >
      <html lang="en">
        <body
          className={cn(
            "min-h-screen bg-background font-sans antialiased",
            fontSans.variable
          )}
        >
          <main className="h-screen w-screen flex">
            <Sidebar />
            <div className="h-full flex-1 relative"> {children}</div>
            <DefaultKBar />
            <Analytics />
          </main>
          <Toaster />
        </body>
      </html>
    </ClerkProvider>
  );
}
