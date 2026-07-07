import { useCallback, useRef, useState } from "react";
import Webcam from "react-webcam";

export const useCamera = () => {
  const webcamRef = useRef<Webcam>(null);
  const [lastCaptureUrl, setLastCaptureUrl] = useState<string>();

  const capture = useCallback(async (): Promise<Blob | undefined> => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) return undefined;
    setLastCaptureUrl(imageSrc);
    const response = await fetch(imageSrc);
    return response.blob();
  }, []);

  return { webcamRef, capture, lastCaptureUrl };
};
