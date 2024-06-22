import { Button } from "@/components/ui/button";

export default function Footer() {
  return (
    <footer className="flex justify-between w-full px-4 mt-10 items-center">
      <Button variant="outline">Email</Button>
      <div className="space-x-4 flex">
        <Button variant="outline">Discord</Button>
        <Button variant="outline">GitHub</Button>
      </div>
    </footer>
  );
}
