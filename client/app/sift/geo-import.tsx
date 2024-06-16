import { useState } from "react";
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
  DialogClose,
} from "@/components/ui/dialog";
import { FileUploader } from "react-drag-drop-files";
import { importData } from "./inout";
import { TableItem, useSift } from "./siftStore";
import { MiniDisplayTable } from "./table";
import { FileInput } from "lucide-react";

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
  const [results, setResults] = useState<TableItem[] | null>(null);
  const { addItems } = useSift();

  const handleFileUpload = (file: File) => {
    if (file) {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        const result = e.target?.result as string;
        processFileContent(result, file.name);
      };
      reader.readAsText(file);
    }
  };

  const processFileContent = (content: string, fileName: string) => {
    const data = tryParse(content, fileName);
    setResults(data);
  };

  const doImport = () => {
    addItems(results!);
    setResults(null);
    setOpen(false);
  };

  // TODO: make wider

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
      <FmtInfo />
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
        <MiniDisplayTable data={results} />
      )}
      {results && (
        <p>
          Loaded <span className="font-bold">{results.length}</span> Records
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

function FmtInfo() {
  return (
    <Collapsible className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-950">
      <CollapsibleTrigger asChild>
        <Button className="w-full justify-between" variant="ghost">
          <div className="flex items-center gap-2">
            <InfoIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
            <span className="font-medium">File Format Specification</span>
          </div>
          <ChevronDownIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-4 p-4">
        <div>
          <h3 className="text-md font-medium">CSV (Comma-Separated Values)</h3>
          <p className="text-gray-500 text-sm dark:text-gray-400">
            CSV files are a simple and widely-used format for storing tabular
            data. They are commonly used for spreadsheets, databases, and data
            exchange.
          </p>
        </div>
        <div>
          <h3 className="text-md font-medium">
            JSON (JavaScript Object Notation)
          </h3>
          <p className="text-gray-500 text-sm dark:text-gray-400">
            JSON is a lightweight data-interchange format that is easy for
            humans to read and write, and easy for machines to parse and
            generate. It is often used for transmitting data between a server
            and web application.
          </p>
        </div>
        <div>
          <h3 className="text-md font-medium">GeoJSON</h3>
          <p className="text-gray-500 text-sm dark:text-gray-400">
            GeoJSON is an open standard format for encoding a variety of
            geographic data structures. It is often used for mapping and
            geographic information systems (GIS) applications.
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
