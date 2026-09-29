import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Cluster Manager — GPU overview",
  description: "Local GPU fleet and project checkpoint dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
