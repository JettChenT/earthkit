"use client";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { login } from "./actions";
import { SiGithub, SiGoogle } from "@icons-pack/react-simple-icons";

export default function LoginForm() {
  return (
    <div className="w-full h-screen">
      <Card className="w-full max-w-sm m-auto mt-12">
        <CardHeader>
          <CardTitle className="text-2xl">Login or Sign Up</CardTitle>
          <CardDescription>
            Log in with any of the supporting providers, an account will be
            created for you if you don't have one.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <Button onClick={() => login("github")}>
            <SiGithub size={18} className="mr-2" /> Log in with Github
          </Button>
          <Button onClick={() => login("google")}>
            <SiGoogle size={18} className="mr-2" /> Log in with Google
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
