"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { useKy } from "@/lib/api-client/api";
import { HTTPError } from "ky";

export default function Playground() {
  const [isRunning, setIsRunning] = useState(false);
  let getKy = useKy();

  return (
    <div className="container mt-10">
      <Button
        onClick={async () => {
          setIsRunning(true);
          let ky = await getKy();
          ky.get("test/echo-user")
            .then((res) => {
              res.json().then((data) => {
                console.log(data);
                setIsRunning(false);
              });
            })
            .catch(async (e) => {
              if (e instanceof HTTPError) {
                const { response } = e;
                if (response && response.body) {
                  console.log(response.status);
                  console.log(await response.json());
                }
                setIsRunning(false);
              }
            });
        }}
      >
        {isRunning ? "Running..." : "Run"}
      </Button>
    </div>
  );
}
