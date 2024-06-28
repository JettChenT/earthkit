import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  CollapsibleTrigger,
  CollapsibleContent,
  Collapsible,
} from "@/components/ui/collapsible";
import {
  DialogTitle,
  DialogDescription,
  DialogHeader,
  DialogContent,
} from "@/components/ui/dialog";
import { FileUploader } from "react-drag-drop-files";
import { TableEncapsulation, importData } from "./inout";
import { useSift } from "./siftStore";
import { MiniDisplayTable } from "./table";
import { compileColDefs, defaultColDefs, defaultCols, mergeCols } from "./cols";
import { toast } from "sonner";
import * as Sentry from "@sentry/browser";

function tryParse(content: string, fileName: string) {
  if (fileName.endsWith(".csv")) {
    return importData(content, "csv");
  }
  const peek = JSON.parse(content);
  if (peek.type === "FeatureCollection" || fileName.endsWith(".geojson")) {
    return importData(content, "geojson");
  }
  return importData(content, "json");
}

export function GeoImport({ setOpen }: { setOpen: (open: boolean) => void }) {
  const [results, setResults] = useState<TableEncapsulation | null>(null);
  const [fmtOpen, fmtSetOpen] = useState(false);
  const { tableImport } = useSift();
  let miniColDef = useMemo(() => {
    if (results) {
      return compileColDefs(mergeCols(defaultCols, results.cols));
    }
    return defaultColDefs;
  }, [results]);

  const handleFileUpload = (file: File) => {
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const result = e.target?.result as string;
        try {
          processFileContent(result, file.name);
        } catch (e) {
          Sentry.captureException(e, { level: "debug" });
          toast.error(
            "Invalid Format! Only valid CSV, JSON, and GeoJSON exports are supported."
          );
        }
      };
      reader.readAsText(file);
    }
  };

  const processFileContent = (content: string, fileName: string) => {
    const data = tryParse(content, fileName);
    setResults(data);
  };

  const doImport = () => {
    tableImport(results!);
    setResults(null);
    setOpen(false);
  };

  // TODO: make wider

  useEffect(() => {
    if (results !== null) {
      fmtSetOpen(false);
    }
  }, [results]);

  return (
    <DialogContent>
      <DialogHeader>
        <DialogTitle className="text-2xl">
          Import Points and Coordinates
        </DialogTitle>
        <DialogDescription>
          Choose a CSV, JSON, or GeoJSON file to upload.
        </DialogDescription>
      </DialogHeader>
      <FmtInfo
        open={fmtOpen}
        setOpen={fmtSetOpen}
        disabled={results !== null}
      />
      {results === null ? (
        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-800 dark:bg-gray-950">
          <FileUploader
            handleChange={handleFileUpload}
            name="file"
            accept="csv, json, geojson"
          >
            <div className="flex h-40 items-center justify-center rounded-lg border-2 border-dashed border-gray-300 bg-gray-50 hover:bg-gray-100 dark:border-gray-700 dark:bg-gray-900 dark:hover:bg-gray-800 hover:cursor-pointer">
              <div className="space-y-2 text-center">
                <UploadIcon className="mx-auto h-8 w-8 text-gray-400 hover:text-gray-500" />
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Drag and drop a file or click to browse
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  CSV, JSON, or GeoJSON files up to 10MB
                </p>
              </div>
            </div>
          </FileUploader>
        </div>
      ) : (
        <MiniDisplayTable data={results.items} columns={miniColDef} />
      )}
      {results && (
        <p>
          Loaded <span className="font-bold">{results.items.length}</span>{" "}
          Records
        </p>
      )}
      {results !== null && (
        <Button
          variant="outline"
          onMouseDown={() => {
            setResults(null);
          }}
        >
          Cancel
        </Button>
      )}
      <Button disabled={results === null} onMouseDown={doImport}>
        Import
      </Button>
    </DialogContent>
  );
}

function FmtInfo({
  open,
  setOpen,
  disabled,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <Collapsible
      open={open}
      onOpenChange={setOpen}
      className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-950"
    >
      <CollapsibleTrigger asChild>
        <Button
          className="w-full justify-between"
          variant="ghost"
          disabled={disabled}
        >
          <div className="flex items-center gap-2">
            <InfoIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            <span className="font-medium">File Format Specification</span>
          </div>
          <ChevronDownIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-4 p-4">
        <div>
          <h3 className="text-md font-medium">CSV</h3>
          <p className="text-gray-500 text-sm dark:text-gray-4000">
            CSV files require the longitude data to be present in a column named
            either &quot;lon&quot;, &quot;lng&quot;, or &quot;longitude&quot;.
            The latitude data must be in a column named &quot;lat&quot; or
            &quot;latitude&quot;. Other column names will also be imported as
            auxiliary features.
          </p>
        </div>
        <div>
          <h3 className="text-md font-medium">GeoJSON</h3>
          <p className="text-gray-500 text-sm dark:text-gray-4000">
            In general Feature Collections of Points should just work. If you
            have a Feature Collection of Polygons or Lines, we will try to
            coerce that into points by taking the midpoint of each feature.
          </p>
        </div>
        <div>
          <h3 className="text-md font-medium">JSON</h3>
          <p className="text-gray-500 text-sm dark:text-gray-400">
            It is not adviced to use JSON for import if the data has not been
            exported from EarthKit. Please reference the{" "}
            <a
              href="https://github.com/JettChenT/earthkit/blob/main/client/app/sift/inout.ts"
              className="text-blue-500 underline"
            >
              source code{" "}
            </a>
            for the specification. (documentation coming soon)
          </p>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function ChevronDownIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function InfoIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  );
}

function UploadIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}
