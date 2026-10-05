import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OpenCardiac AI — Cardiac MRI Segmentation",
  description: "AI-powered automated cardiac structure segmentation using deep learning U-Net on ACDC MRI data. Segment RV, Myocardium, and LV directly in your browser.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
