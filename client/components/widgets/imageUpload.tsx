"use client";
import { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import { twMerge } from "tailwind-merge";
import { Button } from "@/components/ui/button";
import { Image as ImageICN, X } from "lucide-react";

interface ImageUploadProps {
  onSetImage: (image: string) => void;
  onUploadBegin?: () => void;
  fileTypes?: string[];
  className?: string;
  imgClassName?: string;
  image?: string | null;
  content?: string | null;
}

const fileTypes = ["JPG", "PNG", "GIF"];

const ImageUpload: React.FC<ImageUploadProps> = ({
  onSetImage,
  className,
  imgClassName,
  content,
  image,
}) => {
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
        <div className="text-lg font-bold">{content || "Import Image"}</div>
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
