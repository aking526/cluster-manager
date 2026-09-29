export async function POST() {
  try {
    const response = await fetch(`http://127.0.0.1:${process.env.GPU_TRACKER_API_PORT || "8000"}/api/refresh`, {
      method: "POST",
      cache: "no-store",
    });
    return new Response(response.body, {
      status: response.status,
      headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
    });
  } catch {
    return Response.json({ error: "Python API unavailable" }, { status: 503 });
  }
}
