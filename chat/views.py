from django.http import JsonResponse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import httpx

def health(request):
    return JsonResponse({"status": "ok", "app": "synin", "version": 1})

@api_view(["GET"])
def list_models(request):
    """Proxy to LM Studio /v1/models (handy for debugging)."""
    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    url = f"{settings.LM_STUDIO_BASE_URL}/models"
    try:
        with httpx.Client(timeout=60) as client:
            r = client.get(url, headers=headers)
        r.raise_for_status()
        return Response(r.json())
    except httpx.HTTPError as e:
        return Response({"error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

@api_view(["POST"])
def chat_completion(request):
    """
    Body (minimal):
    {
      "messages": [{"role": "user", "content": "Say hello from Synin."}]
    }

    Optional fields you can include:
    {
      "model": "override-model-id",
      "temperature": 0.2,
      "max_tokens": 512,
      "top_p": 1.0
    }
    """
    data = request.data or {}

    messages     = data.get("messages", [])
    model        = data.get("model") or settings.DEFAULT_MODEL_ID
    temperature  = float(data.get("temperature", 0.2))
    max_tokens   = data.get("max_tokens")    # can be None
    top_p        = data.get("top_p", 1.0)

    if not messages:
        return Response({"error": "messages is required"}, status=400)
    if not model:
        return Response({"error": "model is required (set DEFAULT_MODEL_ID in .env or pass 'model')"}, status=400)

    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    url = f"{settings.LM_STUDIO_BASE_URL}/chat/completions"

    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        lm = r.json()

        # Extract assistant content safely
        content = ""
        try:
            content = lm.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        except Exception:
            pass

        return Response({
            "ok": True,
            "model": model,
            "assistant": content.strip(),
            "raw": lm,  # keep for now for debugging; we can remove later
        })
    except httpx.HTTPError as e:
        return Response({"ok": False, "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)