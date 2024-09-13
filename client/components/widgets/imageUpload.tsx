"use client";
import { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import { twMerge } from "tailwind-merge";
import { Button } from "@/components/ui/button";
import { Image as ImageICN, X, LoaderCircleIcon } from "lucide-react";
import { type PutBlobResult } from "@vercel/blob";

interface ImageUploadProps {
  onSetImage: (image: string) => void;
  onUploadBegin?: () => void;
  fileTypes?: string[];
  className?: string;
  imgClassName?: string;
  uploaderClassname?: string;
  image?: string | null;
  content?: string | null;
  id?: string;
}

const fileTypes = ["JPG", "PNG", "GIF", "JPEG"];

const ImageUpload: React.FC<ImageUploadProps> = ({
  id,
  onSetImage,
  className,
  imgClassName,
  uploaderClassname,
  content,
  image,
  onUploadBegin,
}) => {
  const [imgCache, setImgCache] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (file: File) => {
    console.log("file", file);
    if (onUploadBegin) onUploadBegin();
    setUploading(true);
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      setImgCache(result);
    };

    try {
      const response = await fetch(
        `/api/image/upload?type=${file.type.split("/")[1]}`,
        {
          method: "POST",
          body: file,
        }
      );

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      console.log("response", response);
      const blob: PutBlobResult = await response.json();
      onSetImage(blob.url);
    } catch (error) {
      console.error("Error uploading file:", error);
      // Handle error (e.g., show error message to user)
    } finally {
      setUploading(false);
    }
  };

  const handleCancel = () => {
    setImgCache(null);
    onSetImage("");
  };

  return image ? (
    <img
      src={imgCache || image}
      className={twMerge("rounded-md", imgClassName)}
    />
  ) : (
    <FileUploader handleChange={handleFileUpload} name="file" types={fileTypes}>
      <div
        className={twMerge(
          "w-full h-32 bg-slate-300 bg-opacity-20 rounded-md flex items-center justify-center hover:bg-slate-300 hover:bg-opacity-30 border-dashed border-2 border-slate-200 hover:cursor-pointer",
          className
        )}
      >
        {uploading ? (
          <div className="flex items-center">
            <LoaderCircleIcon className="size-4 animate-spin mr-2" />
            <span>Uploading...</span>
          </div>
        ) : (
          <div className="text-lg font-bold">{content || "Import Image"}</div>
        )}
      </div>
    </FileUploader>
  );
};

export function CancellableImage({
  image,
  onCancel,
  className,
}: {
  image: string;
  onCancel: () => void;
  className?: string;
}) {
  return (
    <div className={twMerge("relative group", className)}>
      <img
        src={image}
        alt="Uploaded Image"
        className={"size-16 rounded-md object-cover"}
      />
      <button
        onClick={onCancel}
        className="absolute -right-1 -top-1 size-5 bg-white text-slate-400 border-1 border-black rounded-full flex items-center justify-center shadow-lg hover:bg-slate-100 transition duration-200 opacity-0 group-hover:opacity-100"
      >
        <X className="size-3" />
      </button>
    </div>
  );
}

export const ImageUploadButton: React.FC<{
  onSetImage: (image: string) => void;
  children?: React.ReactNode;
}> = ({ onSetImage, children }) => {
  const [imgCache, setImgCache] = useState<string | null>(null);

  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      setImgCache(result);
      onSetImage(result);
    };
  };

  return (
    <div className="inline-block">
      <FileUploader
        handleChange={handleFileUpload}
        name="file"
        types={fileTypes}
      >
        {children || (
          <Button
            variant="ghost"
            size="sm"
            className="text-xs text-secondary-foreground px-1"
            type="button"
          >
            <ImageICN className="size-3 inline-block mr-1" /> Image
          </Button>
        )}
      </FileUploader>
    </div>
  );
};

export default ImageUpload;
