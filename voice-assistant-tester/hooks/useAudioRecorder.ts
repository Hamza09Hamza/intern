"use client";

import { useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const { toast } = useToast();

  const startRecording = async (
    chunkSize = 1000,
    streamingMode = false,
    onDataAvailable?: (chunk: Blob) => void,
    onStop?: (audioBlob: Blob) => void
  ) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];

      // Try different MIME types in order of preference
      let mimeType = "";
      const supportedTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus",
        "audio/wav",
      ];

      for (const type of supportedTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type;
          break;
        }
      }

      if (!mimeType) {
        console.warn("No supported audio MIME type found, using default");
      }

      const options = mimeType ? { mimeType } : {};
      recorderRef.current = new MediaRecorder(stream, options);

      recorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
          console.log(`Captured chunk: ${event.data.size} bytes`);

          if (streamingMode && onDataAvailable) {
            onDataAvailable(event.data);
          }
        }
      };

      recorderRef.current.onstop = () => {
        // Create blob with the recorded MIME type
        const audioBlob = new Blob(chunksRef.current, {
          type: mimeType || "audio/webm",
        });
        console.log(
          `Final audio blob: ${audioBlob.size} bytes, type: ${audioBlob.type}`
        );

        if (onStop) {
          onStop(audioBlob);
        }

        chunksRef.current = [];
      };

      recorderRef.current.start(chunkSize);
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

      console.log(`Recording started with MIME type: ${mimeType || "default"}`);
      return mimeType || "audio/webm";
    } catch (error) {
      console.error("Error starting recording:", error);
      toast({
        title: "Recording Error",
        description: "Could not access microphone. Please check permissions.",
        variant: "destructive",
      });
      throw error;
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;

    // Stop the MediaRecorder
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }

    // Stop all tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    // Clear timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setIsRecording(false);
    console.log("Recording stopped");
  };

  return {
    isRecording,
    recordingTime,
    startRecording,
    stopRecording,
  };
}
