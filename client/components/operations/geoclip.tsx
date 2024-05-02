import { FileUploader } from "react-drag-drop-files";
import { useState } from "react";
import { useStore } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/lib/constants";
import { Loader2 } from "lucide-react";
import { ProbScatterLayer } from "@/lib/layers";

const fileTypes = ["JPG", "PNG", "GIF"];

export default function GeoCLIPPanel() {
  const [image, setImage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const { setLayers } = useStore();

  const onUpload = (file: File) => {
    console.log("on drop!");
    console.log(file);
    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
      setImage(reader.result as string);
    };
  };

  const onInference = () => {
    setIsRunning(true);
    fetch(`${API_URL}/geoclip`, {
      method: "POST",
      body: JSON.stringify({
        image_url: image,
      }),
      headers: {
        "Content-Type": "application/json",
      },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data);
        const max_conf = Math.max(...data.map((d: any) => d.aux.pred));
        const adjusted_data = data.map((d: any) => {
          d.aux.conf = Math.sqrt(d.aux.pred / max_conf);
          return d;
        });
        const layer = {
          id: "geoclip_pred",
          name: "geoclip prediction",
          type: "prob_scatter",
          coords: adjusted_data,
          key: "conf",
        };
        setLayers([layer as ProbScatterLayer]);
        setIsRunning(false);
      });
  };

  const onCancel = () => {
    setIsRunning(false);
    setImage(null);
  };

  return (
    <div>
      <p className="prose prose-sm leading-5 mb-2">
        <h3>GeoCLIP Geoestimation</h3>
        <a
          className="text-primary"
          href="https://github.com/VicenteVivan/geo-clip"
        >
          GeoCLIP
        </a>{" "}
        predicts the location of an image based on its visual features.
      </p>
      {image ? (
        <img className="rounded-md" src={image} />
      ) : (
        <FileUploader handleChange={onUpload} name="file" types={fileTypes}>
          <div className="w-full h-32 bg-slate-300 bg-opacity-20 rounded-md flex items-center justify-center hover:bg-slate-300 hover:bg-opacity-30 border-dashed border-2 border-slate-200 hover:cursor-pointer">
            <div className="text-lg font-bold">Import Image</div>
          </div>
        </FileUploader>
      )}
      <div>
        <Button
          className={`mt-3`}
          disabled={!image || isRunning}
          onClick={onInference}
        >
          {isRunning ? <Loader2 className="animate-spin mr-2" /> : null}
          {isRunning ? "Predicting..." : "Predict"}
        </Button>
        <Button
          className={`mt-3 ml-2`}
          variant="secondary"
          onClick={onCancel}
          disabled={!image}
        >
          Cancel
        </Button>
      </div>
    </div>
  );
}
