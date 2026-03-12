"use client";

import { useMemo, useState } from "react";

type PredictionResponse = {
  predicted_class: number;
  label: string;
  confidence: number;
  lesion_area_ratio: number;
  mask_base64: string;
  overlay_base64: string;
};

function base64ToImageSrc(base64: string) {
  return `data:image/png;base64,${base64}`;
}

function downloadBase64Image(base64: string, filename: string) {
  const link = document.createElement("a");
  link.href = `data:image/png;base64,${base64}`;
  link.download = filename;
  link.click();
}

function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();

  URL.revokeObjectURL(url);
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const diagnosisColor = useMemo(() => {
    if (!result) return "text-slate-900";
    return result.label === "Benign" ? "text-emerald-700" : "text-red-700";
  }, [result]);

  const handleFileChange = (selectedFile: File | null) => {
    setResult(null);
    setError("");

    if (!selectedFile) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!allowedTypes.includes(selectedFile.type)) {
      setError("Please upload a valid JPG, JPEG, or PNG image.");
      setFile(null);
      setPreviewUrl(null);
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError("");
  };

  const handleAnalyze = async () => {
    if (!apiUrl) {
      setError("API URL is missing. Please check frontend/.env.local");
      return;
    }

    if (!file) {
      setError("Please select an image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      });

      let data: unknown = null;
      try {
        data = await response.json();
      } catch {
        throw new Error("Server returned an invalid response.");
      }

      if (!response.ok) {
        const message =
          typeof data === "object" &&
          data !== null &&
          "detail" in data &&
          typeof (data as { detail?: unknown }).detail === "string"
            ? (data as { detail: string }).detail
            : "Failed to analyze image.";
        throw new Error(message);
      }

      setResult(data as PredictionResponse);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the image.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <section className="mb-8 rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
          <p className="mb-2 text-sm font-semibold uppercase tracking-[0.25em] text-blue-700">
            Dermascan AI
          </p>
          <h1 className="text-4xl font-bold tracking-tight">
            AI skin lesion analysis platform
          </h1>
          <p className="mt-3 max-w-3xl text-slate-600">
            Upload a dermoscopic image and get AI-based lesion classification,
            confidence score, lesion area ratio, and visual mask overlays.
          </p>
        </section>

        <div className="grid gap-8 xl:grid-cols-[420px_minmax(0,1fr)]">
          <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-2xl font-semibold">Upload image</h2>
            <p className="mt-2 text-sm text-slate-600">
              Supported formats: JPG, JPEG, PNG
            </p>

            <label className="mt-6 flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 py-12 text-center transition hover:border-blue-500 hover:bg-blue-50">
              <span className="text-lg font-medium">Choose a skin lesion image</span>
              <span className="mt-2 text-sm text-slate-500">
                Click here to browse from your device
              </span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                className="hidden"
                onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
              />
            </label>

            <div className="mt-6 grid gap-3 sm:grid-cols-2">
              <button
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="inline-flex w-full items-center justify-center rounded-2xl bg-blue-700 px-5 py-3 text-base font-semibold text-white transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {loading ? "Analyzing..." : "Analyze image"}
              </button>

              <button
                onClick={handleClear}
                disabled={loading}
                className="inline-flex w-full items-center justify-center rounded-2xl border border-slate-300 bg-white px-5 py-3 text-base font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-60"
              >
                Clear
              </button>
            </div>

            {loading && (
              <div className="mt-4 rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-700">
                The model is analyzing the image. Please wait...
              </div>
            )}

            {error && (
              <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}

            <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-medium text-slate-500">Clinical note</p>
              <p className="mt-2 text-sm text-slate-600">
                This application is a research prototype and must not be used
                as a substitute for professional medical diagnosis.
              </p>
            </div>
          </section>

          <section className="space-y-8">
            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <h2 className="text-2xl font-semibold">Visual analysis</h2>

                {result && (
                  <div className="flex flex-wrap gap-2">
                    <button
                      onClick={() =>
                        downloadBase64Image(result.mask_base64, "dermascan-mask.png")
                      }
                      className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    >
                      Download Mask
                    </button>

                    <button
                      onClick={() =>
                        downloadBase64Image(
                          result.overlay_base64,
                          "dermascan-overlay.png"
                        )
                      }
                      className="rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-100"
                    >
                      Download Overlay
                    </button>

                    <button
                      onClick={() => downloadJson(result, "dermascan-result.json")}
                      className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800"
                    >
                      Download JSON
                    </button>
                  </div>
                )}
              </div>

              <div className="mt-6 grid gap-6 md:grid-cols-3">
                <div>
                  <p className="mb-3 text-sm font-medium text-slate-500">Original image</p>
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-100">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="Uploaded preview"
                        className="h-[320px] w-full object-contain bg-white"
                      />
                    ) : (
                      <div className="flex h-[320px] items-center justify-center text-slate-400">
                        No image selected
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <p className="mb-3 text-sm font-medium text-slate-500">Predicted mask</p>
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-100">
                    {result?.mask_base64 ? (
                      <img
                        src={base64ToImageSrc(result.mask_base64)}
                        alt="Predicted mask"
                        className="h-[320px] w-full object-contain bg-white"
                      />
                    ) : (
                      <div className="flex h-[320px] items-center justify-center text-slate-400">
                        No mask available yet
                      </div>
                    )}
                  </div>
                </div>

                <div>
                  <p className="mb-3 text-sm font-medium text-slate-500">Overlay</p>
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-100">
                    {result?.overlay_base64 ? (
                      <img
                        src={base64ToImageSrc(result.overlay_base64)}
                        alt="Overlay result"
                        className="h-[320px] w-full object-contain bg-white"
                      />
                    ) : (
                      <div className="flex h-[320px] items-center justify-center text-slate-400">
                        No overlay available yet
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
              <h2 className="text-2xl font-semibold">Analysis result</h2>

              {!result ? (
                <div className="mt-6 rounded-2xl border border-slate-200 bg-slate-50 px-5 py-10 text-center text-slate-500">
                  Upload an image and click analyze to see the prediction.
                </div>
              ) : (
                <>
                  <div className="mt-6 grid gap-4 md:grid-cols-3">
                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                      <p className="text-sm font-medium text-slate-500">Diagnosis</p>
                      <p className={`mt-3 text-3xl font-bold ${diagnosisColor}`}>
                        {result.label}
                      </p>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                      <p className="text-sm font-medium text-slate-500">Confidence</p>
                      <p className="mt-3 text-3xl font-bold text-slate-900">
                        {(result.confidence * 100).toFixed(2)}%
                      </p>
                    </div>

                    <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5">
                      <p className="text-sm font-medium text-slate-500">
                        Lesion Area Ratio
                      </p>
                      <p className="mt-3 text-3xl font-bold text-slate-900">
                        {(result.lesion_area_ratio * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>

                  <div
                    className={`mt-6 rounded-2xl border px-4 py-4 text-sm ${
                      result.label === "Benign"
                        ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                        : "border-red-200 bg-red-50 text-red-800"
                    }`}
                  >
                    {result.label === "Benign"
                      ? "The model predicts this lesion as benign. Clinical confirmation is still recommended."
                      : "The model predicts this lesion as malignant. Professional medical review is strongly recommended."}
                  </div>
                </>
              )}
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}