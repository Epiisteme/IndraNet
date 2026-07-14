import { Camera, ImageUp } from "lucide-react";
import type { ChangeEvent } from "react";
import Webcam from "react-webcam";

import { useCamera } from "../hooks/useCamera";

interface IrisCaptureProps {
  onCapture: (imageBlob: Blob) => void;
  disabled?: boolean;
  label?: string;
}

const videoConstraints: MediaTrackConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

export function IrisCapture({ onCapture, disabled, label = "Iris sample" }: IrisCaptureProps) {
  const { webcamRef, capture, lastCaptureUrl } = useCamera();

  const handleCameraCapture = async () => {
    const blob = await capture();
    if (blob) {
      onCapture(blob);
    }
  };

  const handleFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onCapture(file);
    }
  };

  return (
    <div className="capture-shell">
      <h3 className="capture-title">{label}</h3>
      <div className="camera-frame">
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          className="camera-video"
        />
        <div className="reticle" aria-hidden="true" />
        {lastCaptureUrl ? <img src={lastCaptureUrl} alt="" className="last-capture" /> : null}
      </div>
      <div className="capture-actions">
        <button className="icon-button primary" onClick={handleCameraCapture} disabled={disabled} title="Capture iris">
          <Camera size={18} />
          <span>Capture sample</span>
        </button>
        <label className="icon-button" title="Upload iris image">
          <ImageUp size={18} />
          <span>Upload image</span>
          <input type="file" accept="image/*" onChange={handleFile} disabled={disabled} />
        </label>
      </div>
      <p className="help-text">
        Biometric proof is derived from this image for matching. Use a clear, well-lit eye image; do not upload unrelated
        personal documents.
      </p>
    </div>
  );
}
