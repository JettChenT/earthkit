"use client";
import { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import { twMerge } from "tailwind-merge";

interface ImageUploadProps {
  onSetImage: (image: string) => void;
  onUploadBegin?: () => void;
  fileTypes?: string[];
  className?: string;
  image?: string | null;
}

const fileTypes = ["JPG", "PNG", "GIF"];

const ImageUpload: React.FC<ImageUploadProps> = ({
  onSetImage,
  className,
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
    <img src={imgCache!} className="rounded-md" />
  ) : (
    <FileUploader handleChange={handleFileUpload} name="file" types={fileTypes}>
      <div
        className={twMerge(
          "w-full h-32 bg-slate-300 bg-opacity-20 rounded-md flex items-center justify-center hover:bg-slate-300 hover:bg-opacity-30 border-dashed border-2 border-slate-200 hover:cursor-pointer",
          className
        )}
      >
        <div className="text-lg font-bold">Import Image</div>
      </div>
    </FileUploader>
  );
};

export default ImageUpload;
