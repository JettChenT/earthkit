import { createClient } from "@/lib/supabase/client";
import { User } from "@supabase/supabase-js";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Skeleton } from "./ui/skeleton";
import Image from "next/image";
import { Button } from "./ui/button";
import { LogOut, LogIn } from "lucide-react";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function Profile() {
  const supabase = createClient();
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    supabase.auth
      .getSession()
      .then((session) => {
        console.log(session, session.data.session?.user);
        setUser(session.data.session?.user ?? null);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching user:", error);
      });
  }, []);
  return (
    <div className="flex items-center gap-3 rounded-lg px-7 py-2 text-muted-foreground hover:text-primary">
      {loading ? (
        <div className="flex items-center space-x-4">
          <Skeleton className="h-7 w-7 rounded-full" />
          <div className="space-y-2">
            <Skeleton className="h-2 w-[40px]" />
            <Skeleton className="h-2 w-[80px]" />
          </div>
        </div>
      ) : user ? (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <div className="flex items-center justify-start gap-2 hover:bg-secondary rounded-lg p-2 w-full hover:cursor-pointer">
              {user.user_metadata.avatar_url ? (
                <Image
                  src={user.user_metadata.avatar_url}
                  alt="Profile Image"
                  width={25}
                  height={25}
                  className="rounded-full w-7 h-7"
                />
              ) : (
                <></>
              )}
              <div>{user.user_metadata.name ?? user.email}</div>
            </div>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-32">
            <DropdownMenuItem
              onClick={() => {
                supabase.auth.signOut();
                setUser(null);
                toast.success("Logged out");
              }}
            >
              <LogOut className="w-4 h-4 mr-2" />
              <span>Log Out</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ) : (
        <Link href="/login" className="flex items-center gap-2">
          <LogIn className="w-4 h-4 mr-2" />
          <span>Sign In</span>
        </Link>
      )}
    </div>
  );
}
