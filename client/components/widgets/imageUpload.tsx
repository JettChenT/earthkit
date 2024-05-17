"use client";
import { UploadDropzone } from "@/lib/uploadthing";
import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import { twMerge } from "tailwind-merge";
import { Image } from "lucide-react";

interface ImageUploadProps {
  onSetImage: (image: string) => void;
  onUploadBegin?: () => void;
  fileTypes?: string[];
  className?: string;
  image?: string | null;
}

const ImageUpload: React.FC<ImageUploadProps> = ({
  onSetImage,
  className,
  onUploadBegin,
  image,
}) => {
  const [imgCache, setImgCache] = useState<string | null>(null);
  const handleFileUpload = (file: File) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      setImgCache(reader.result as string);
    };
  };

  return image ? (
    <img src={imgCache!} className="rounded-md" />
  ) : (
    <>
      <UploadDropzone
        endpoint="imageUploader"
        className={twMerge("p-5", className)}
        content={{
          label: "Choose Image or Drop here",
          uploadIcon: <Image className="size-10 text-gray-700" />,
        }}
        config={{ mode: "auto" }}
        onUploadBegin={onUploadBegin}
        onBeforeUploadBegin={(files) => {
          const reader = new FileReader();
          reader.readAsDataURL(files[0]);
          reader.onload = () => {
            setImgCache(reader.result as string);
          };
          return files;
        }}
        onClientUploadComplete={(res) => {
          console.log("Files: ", res);
          onSetImage(res[0].url);
        }}
      />
    </>
  );
};

export default ImageUpload;
