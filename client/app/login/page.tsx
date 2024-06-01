import { Button } from "@/components/ui/button";
import { login } from "./actions";

export default function LoginPage() {
  return (
    <form>
      <Button formAction={login}>Log in with Github</Button>
    </form>
  );
}
