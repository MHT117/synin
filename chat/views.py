from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Conversation, Message
from django.db import transaction
from django.utils.text import Truncator
import time, logging, json

import json
import httpx

log = logging.getLogger("synin")

#validator helper and use consistend timeouts
def _validate_messages(msgs):
    if not isinstance(msgs, list) or not msgs:
        return "messages must be a non-empty list"
    if len(msgs) > settings.HISTORY_MAX_TURNS:
        return f"too many turns (>{settings.HISTORY_MAX_TURNS})"
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role not in {"system", "user", "assistant"}:
            return "invalid role"
        if not isinstance(content, str):
            return "content must be string"
        if len(content) > settings.MESSAGE_MAX_CHARS:
            return f"message too long (>{settings.MESSAGE_MAX_CHARS} chars)"
    return None
#conversation 
def _get_or_create_conversation(conv_id: str | None, session_key: str | None) -> Conversation:
    if conv_id:
        try:
            return Conversation.objects.get(id=conv_id)
        except Conversation.DoesNotExist:
            pass
    return Conversation.objects.create(session_key=session_key or "")

def _maybe_set_title(conv: Conversation):
    # set a simple title from first user message if empty
    if not conv.title:
        first = conv.messages.filter(role="user").first()
        if first:
            conv.title = (first.content[:60] + "…") if len(first.content) > 60 else first.content
            conv.save(update_fields=["title"])

@api_view(["GET"])
def get_conversation(request, conv_id):
    try:
        conv = Conversation.objects.get(id=conv_id)
    except Conversation.DoesNotExist:
        return Response({"error": "not found"}, status=404)
    msgs = [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in conv.messages.all()]
    return Response({"id": str(conv.id), "title": conv.title, "messages": msgs})

@api_view(["POST"])
def clear_conversation(request, conv_id):
    try:
        conv = Conversation.objects.get(id=conv_id)
    except Conversation.DoesNotExist:
        return Response({"error": "not found"}, status=404)
    conv.messages.all().delete()
    conv.title = ""
    conv.save(update_fields=["title"])
    return Response({"ok": True})


# ---------- Web page ----------
def index(request):
    # Looks for templates/index.html because settings.TEMPLATES includes BASE_DIR / 'templates'
    return render(request, "index.html")


# ---------- Simple health check ----------
def health(request):
    return JsonResponse({"status": "ok", "app": "synin", "version": 1})


# ---------- API: list models (proxy to LM Studio) ----------
@csrf_exempt
@api_view(["GET"])
def list_models(request):
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


# ---------- API: non-streaming chat ----------
@csrf_exempt
@api_view(["POST"])
def chat_completion(request):
    data = request.data or {}
    messages_in   = data.get("messages", [])
    model         = data.get("model") or settings.DEFAULT_MODEL_ID
    temperature   = float(data.get("temperature", 0.2))
    top_p         = data.get("top_p", 1.0)
    max_tokens    = data.get("max_tokens")
    conversation_id = data.get("conversation_id")
    session_key   = request.session.session_key or ""

    # Validate
    err = _validate_messages(messages_in)
    if err:
        return Response({"error": err}, status=400)
    if not model:
        return Response({"error": "model is required"}, status=400)

    # Cap max_tokens
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except Exception:
            return Response({"error": "max_tokens must be integer"}, status=400)
        if max_tokens > settings.MAX_TOKENS_LIMIT:
            max_tokens = settings.MAX_TOKENS_LIMIT

    # Persist incoming user/system turns
    with transaction.atomic():
        conv = _get_or_create_conversation(conversation_id, session_key)
        for m in messages_in:
            if m.get("role") in {"system", "user"}:
                Message.objects.create(conversation=conv, role=m["role"], content=m["content"])

    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    payload = {"model": model, "messages": messages_in, "temperature": temperature, "top_p": top_p}
    if max_tokens:
        payload["max_tokens"] = max_tokens

    url = f"{settings.LM_STUDIO_BASE_URL}/chat/completions"

    # httpx timeout object
    timeout = httpx.Timeout(
        connect=settings.CONNECT_TIMEOUT,
        read=settings.REQUEST_TIMEOUT,
        write=settings.REQUEST_TIMEOUT,
        pool=settings.CONNECT_TIMEOUT,
    )

    try:
        t0 = time.time()
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        lm = r.json()
        dt_ms = (time.time() - t0) * 1000.0

        content = lm.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        # Save assistant
        with transaction.atomic():
            Message.objects.create(conversation=conv, role="assistant", content=content.strip())
            _maybe_set_title(conv)

        log.info("LMStudio chat %.0fms | prompt=%s",
                 dt_ms,
                 Truncator(messages_in[-1]["content"]).chars(120))

        return Response({
            "ok": True,
            "model": model,
            "assistant": content.strip(),
            "conversation_id": str(conv.id),
        })

    except httpx.HTTPError as e:
        log.warning("LMStudio error: %s", e)
        return Response({"ok": False, "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)


# ---------- API: streaming chat (SSE-style) ----------
@csrf_exempt
def chat_stream(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    messages_in     = data.get("messages", [])
    model           = data.get("model") or settings.DEFAULT_MODEL_ID
    temperature     = float(data.get("temperature", 0.2))
    top_p           = data.get("top_p", 1.0)
    max_tokens      = data.get("max_tokens")
    conversation_id = data.get("conversation_id")
    session_key     = request.session.session_key or ""

    # Validate
    err = _validate_messages(messages_in)
    if err:
        return JsonResponse({"error": err}, status=400)
    if not model:
        return JsonResponse({"error": "model is required"}, status=400)

    # Cap max_tokens
    if max_tokens is not None:
        try:
            max_tokens = int(max_tokens)
        except Exception:
            return JsonResponse({"error": "max_tokens must be integer"}, status=400)
        if max_tokens > settings.MAX_TOKENS_LIMIT:
            max_tokens = settings.MAX_TOKENS_LIMIT

    # Persist incoming user/system turns first
    with transaction.atomic():
        conv = _get_or_create_conversation(conversation_id, session_key)
        for m in messages_in:
            if m.get("role") in {"system", "user"}:
                Message.objects.create(conversation=conv, role=m["role"], content=m["content"])

    headers = {"Content-Type": "application/json"}
    if settings.LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LM_STUDIO_API_KEY}"

    payload = {
        "model": model,
        "messages": messages_in,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens

    url = f"{settings.LM_STUDIO_BASE_URL}/chat/completions"

    def event_stream():
        reply_accum = []
        log.info("LMStudio stream start | prompt=%s",
                 Truncator(messages_in[-1]["content"]).chars(120))
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", url, json=payload, headers=headers) as r:
                    r.raise_for_status()
                    for raw in r.iter_lines():
                        if not raw:
                            continue
                        try:
                            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                        except Exception:
                            text = str(raw)

                        line = text.strip()
                        if line.startswith("data:"):
                            line = line[5:].strip()

                        # collect for DB save
                        try:
                            j = json.loads(line)
                            delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                            if delta:
                                reply_accum.append(delta)
                        except Exception:
                            pass

                        # forward to client with exactly one "data:"
                        yield f"data: {line}\n\n"
        finally:
            full = "".join(reply_accum).strip()
            if full:
                with transaction.atomic():
                    Message.objects.create(conversation=conv, role="assistant", content=full)
                    _maybe_set_title(conv)
                log.info("LMStudio stream end | %d chars", len(full))
            yield "data: [DONE]\n\n"

    resp = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    resp["X-Conversation-Id"] = str(conv.id)
    return resp


