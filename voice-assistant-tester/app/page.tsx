"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { useAudioRecorder } from "../hooks/useAudioRecorder";
import { Mic, MicOff, Settings, Trash2 } from "lucide-react";

interface ServerResponse {
  success: boolean;
  transcription?: string;
  answer?: string;
  audioUrl?: string;
  filename?: string;
  error?: string;
}

export default function VoiceAssistantTester() {
  const [serverUrl, setServerUrl] = useState("http://localhost:8000");
  const [response, setResponse] = useState<ServerResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [history, setHistory] = useState<ServerResponse[]>([]);
  const audioRecorder = useAudioRecorder();
  const { toast } = useToast();

  const sendAudioToServer = async (audioBlob: Blob) => {
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");

      // Step 1: Process audio and get transcription, answer, and audio_id
      const processResponse = await fetch(`${serverUrl}/process_audio`, {
        method: "POST",
        body: formData,
      });

      if (!processResponse.ok) {
        throw new Error(`Server error: ${processResponse.status}`);
      }

      const processResult = await processResponse.json();

      if (!processResult.success) {
        // Handle processing error
        const errorResult: ServerResponse = {
          success: false,
          transcription: processResult.transcription || "",
          answer: processResult.answer || "",
          error: processResult.error || "Processing failed",
        };

        setResponse(errorResult);
        setHistory((prev) => [errorResult, ...prev]);

        toast({
          title: "Processing Error",
          description: errorResult.error,
          variant: "destructive",
        });
        return;
      }

      // Step 2: Get the audio stream using the audio_id
      const audioResponse = await fetch(
        `${serverUrl}/stream_audio/${processResult.audio_id}`
      );

      if (!audioResponse.ok) {
        throw new Error(`Audio streaming error: ${audioResponse.status}`);
      }

      const responseAudioBlob = await audioResponse.blob();
      const audioUrl = URL.createObjectURL(responseAudioBlob);

      const result: ServerResponse = {
        success: true,
        transcription: processResult.transcription,
        answer: processResult.answer,
        audioUrl: audioUrl,
      };

      setResponse(result);
      setHistory((prev) => [result, ...prev]);

      toast({
        title: "Success",
        description: "Audio processed and speech generated",
      });

      // Auto-play the response with better error handling
      try {
        const audio = new Audio(audioUrl);

        // Add error event listener
        audio.addEventListener("error", (e) => {
          console.error("Audio playback error:", e);
          toast({
            title: "Audio Playback Error",
            description:
              "Could not play audio response. You can still use the audio controls.",
            variant: "destructive",
          });
        });

        // Add canplay event listener to ensure audio is ready
        audio.addEventListener("canplay", () => {
          console.log("Audio is ready to play");
        });

        await audio.play();
      } catch (playError) {
        console.error("Auto-play failed:", playError);
        toast({
          title: "Auto-play Failed",
          description: "Audio response ready - click play button to listen",
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      toast({
        title: "Connection Error",
        description: `Failed to connect to server: ${errorMessage}`,
        variant: "destructive",
      });

      setResponse({
        success: false,
        transcription: "",
        answer: "",
        error: errorMessage,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${serverUrl}/health`);

      if (response.ok) {
        toast({
          title: "Server Status",
          description: "Server is running and all services are initialized",
        });
      } else {
        toast({
          title: "Server Status",
          description: "Server is running but services may not be ready",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Server Status",
        description: "Cannot connect to server",
        variant: "destructive",
      });
    }
  };

  const handleStartRecording = async () => {
    try {
      await audioRecorder.startRecording(
        1000, // 1 second chunks
        false, // batch mode
        undefined, // no streaming callback
        sendAudioToServer // callback when recording stops
      );

      toast({
        title: "Recording Started",
        description: "Speak now... Click stop when finished",
      });
    } catch (error) {
      toast({
        title: "Recording Error",
        description: "Could not access microphone",
        variant: "destructive",
      });
    }
  };

  const handleStopRecording = () => {
    audioRecorder.stopRecording();
    toast({
      title: "Recording Stopped",
      description: "Processing audio...",
    });
  };

  const clearHistory = () => {
    // Clean up audio URLs to prevent memory leaks
    history.forEach((item) => {
      if (item.audioUrl) {
        URL.revokeObjectURL(item.audioUrl);
      }
    });

    if (response?.audioUrl) {
      URL.revokeObjectURL(response.audioUrl);
    }

    setHistory([]);
    setResponse(null);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Cleanup audio URLs on unmount
  useEffect(() => {
    return () => {
      history.forEach((item) => {
        if (item.audioUrl) {
          URL.revokeObjectURL(item.audioUrl);
        }
      });

      if (response?.audioUrl) {
        URL.revokeObjectURL(response.audioUrl);
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Voice Assistant Tester
          </h1>
          <p className="text-gray-600">
            Record audio and test your voice assistant server
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          {/* Controls Panel */}
          <Card className="md:col-span-1">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Controls
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Server Configuration */}
              <div className="space-y-2">
                <Label htmlFor="server-url">Server URL</Label>
                <Input
                  id="server-url"
                  value={serverUrl}
                  onChange={(e) => setServerUrl(e.target.value)}
                  placeholder="http://localhost:8000"
                  disabled={audioRecorder.isRecording || isProcessing}
                />
                <Button
                  onClick={checkServerHealth}
                  variant="outline"
                  size="sm"
                  className="w-full bg-transparent"
                  disabled={audioRecorder.isRecording || isProcessing}
                >
                  Check Server Health
                </Button>
              </div>

              {/* Recording Button */}
              <div className="space-y-2">
                <Label>Recording</Label>
                <div className="flex flex-col items-center space-y-2">
                  <Button
                    onClick={
                      audioRecorder.isRecording
                        ? handleStopRecording
                        : handleStartRecording
                    }
                    disabled={isProcessing}
                    size="lg"
                    className={`w-full ${
                      audioRecorder.isRecording
                        ? "bg-red-500 hover:bg-red-600"
                        : "bg-blue-500 hover:bg-blue-600"
                    }`}
                  >
                    {audioRecorder.isRecording ? (
                      <>
                        <MicOff className="w-5 h-5 mr-2" />
                        Stop Recording
                      </>
                    ) : (
                      <>
                        <Mic className="w-5 h-5 mr-2" />
                        Start Recording
                      </>
                    )}
                  </Button>

                  {audioRecorder.isRecording && (
                    <div className="text-center">
                      <div className="text-sm text-gray-600">Recording...</div>
                      <div className="text-lg font-mono text-red-600">
                        {formatTime(audioRecorder.recordingTime)}
                      </div>
                    </div>
                  )}

                  {isProcessing && (
                    <div className="text-center">
                      <div className="text-sm text-blue-600">Processing...</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Clear History */}
              <Button
                onClick={clearHistory}
                variant="outline"
                className="w-full bg-transparent"
                disabled={history.length === 0}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear History
              </Button>
            </CardContent>
          </Card>

          {/* Response Display */}
          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle className="text-lg">Server Response</CardTitle>
            </CardHeader>
            <CardContent>
              {response ? (
                <div className="space-y-4">
                  {/* Current Response */}
                  <div className="border rounded-lg p-4 bg-white">
                    <div className="flex items-center gap-2 mb-2">
                      <div
                        className={`w-3 h-3 rounded-full ${
                          response.success ? "bg-green-500" : "bg-red-500"
                        }`}
                      />
                      <span className="font-medium">
                        {response.success ? "Success" : "Error"}
                      </span>
                    </div>

                    {response.success ? (
                      <>
                        <div className="mb-3">
                          <Label className="text-sm font-medium text-gray-700">
                            Transcription:
                          </Label>
                          <div className="mt-1 p-2 bg-gray-50 rounded border text-sm">
                            {response.transcription || "No speech detected"}
                          </div>
                        </div>

                        <div className="mb-3">
                          <Label className="text-sm font-medium text-gray-700">
                            Assistant Answer:
                          </Label>
                          <div className="mt-1 p-2 bg-green-50 rounded border text-sm">
                            {response.answer || "No response generated"}
                          </div>
                        </div>

                        {response.audioUrl && (
                          <div className="mb-3">
                            <Label className="text-sm font-medium text-gray-700">
                              Audio Response:
                            </Label>
                            <div className="mt-1 p-2 bg-blue-50 rounded border">
                              <audio
                                controls
                                className="w-full"
                                src={response.audioUrl}
                                preload="metadata"
                                onError={(e) => {
                                  console.error("Audio element error:", e);
                                  toast({
                                    title: "Audio Error",
                                    description:
                                      "Could not load audio response",
                                    variant: "destructive",
                                  });
                                }}
                              >
                                Your browser does not support the audio element.
                              </audio>
                              <div className="text-xs text-gray-600 mt-1">
                                Click play to hear the assistant's response
                              </div>
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div>
                        <Label className="text-sm font-medium text-red-700">
                          Error:
                        </Label>
                        <div className="mt-1 p-2 bg-red-50 rounded border text-sm text-red-800">
                          {response.error || "Unknown error occurred"}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* History */}
                  {history.length > 1 && (
                    <div>
                      <Label className="text-sm font-medium text-gray-700 mb-2 block">
                        Previous Responses:
                      </Label>
                      <div className="space-y-2 max-h-64 overflow-y-auto">
                        {history.slice(1).map((item, index) => (
                          <div
                            key={index}
                            className="border rounded p-3 bg-gray-50 text-sm"
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <div
                                className={`w-2 h-2 rounded-full ${
                                  item.success ? "bg-green-400" : "bg-red-400"
                                }`}
                              />
                              <span className="text-xs text-gray-600">
                                Response #{history.length - index - 1}
                              </span>
                            </div>
                            {item.success ? (
                              <>
                                <div className="text-gray-700 mb-1">
                                  <strong>Q:</strong> {item.transcription}
                                </div>
                                <div className="text-green-700 mb-2">
                                  <strong>A:</strong> {item.answer}
                                </div>
                                {item.audioUrl && (
                                  <div className="mb-2">
                                    <audio
                                      controls
                                      className="w-full h-8"
                                      src={item.audioUrl}
                                      preload="metadata"
                                    >
                                      Your browser does not support the audio
                                      element.
                                    </audio>
                                  </div>
                                )}
                              </>
                            ) : (
                              <div className="text-red-700">
                                Error: {item.error}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Mic className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>Click "Start Recording" to begin testing</p>
                  <p className="text-sm mt-2">
                    Make sure your server is running at {serverUrl}
                  </p>
                  <p className="text-xs mt-1 text-gray-400">
                    Server will return audio responses that will auto-play
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
